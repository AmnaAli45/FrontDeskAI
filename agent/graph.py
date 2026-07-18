from datetime import date
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Literal as LiteralType
from langgraph.graph import START, END, StateGraph
from django.conf import settings

from agent.state import State
from agent.tools import (
    check_availability, create_appointment, get_services_list,
    find_appointment, update_appointment, cancel_appointment,
    looks_like_malformed_tool_call, parse_malformed_tool_call, notify_staff
)
from agent.prompts import (
    MESSAGE_CLASSIFICATION_PROMPT_TEMPLATE, BOOKING_SYSTEM_PROMPT,
    RESCHEDULE_SYSTEM_PROMPT, FAQ_SYSTEM_PROMPT,
    ESCALATE_CLASSIFY_PROMPT, GREETING_RESPONSE_PROMPT
)

model = ChatGroq(model="llama-3.3-70b-versatile")

# ------------------------------------------ Classification schema -------------------------------------------------------------

class MessageClassification(BaseModel):
    category: LiteralType["booking", "faq", "reschedule", "other"]
    confidence: LiteralType["low", "high"]

structured_model = model.with_structured_output(MessageClassification, strict=True)

# ----------------------------------------------------- Making Models aware of Tool -----------------------------------------------------

booking_model = model.bind_tools([check_availability, create_appointment, get_services_list])
reschedule_model = model.bind_tools([find_appointment, update_appointment, cancel_appointment])
faq_model = model.bind_tools([get_services_list])

BOOKING_TOOL_MAP = {"check_availability": check_availability, "create_appointment": create_appointment,
                     "get_services_list": get_services_list}
RESCHEDULE_TOOL_MAP = {"find_appointment": find_appointment, "update_appointment": update_appointment,
                        "cancel_appointment": cancel_appointment}
FAQ_TOOL_MAP = {"get_services_list": get_services_list}


# -------------------------------------------- Generic tool-calling loop (reused by booking/reschedule/faq) ---------------------------------

def run_tool_calling_loop(bound_model, tool_map, messages, max_iterations=5):
    """Robust loop: normal tool_calls + malformed <function=...> fallback."""
    response = bound_model.invoke(messages)
    messages.append(response)
    iterations = 0

    while iterations < max_iterations:
        if response.tool_calls:
            for call in response.tool_calls:
                fn = tool_map.get(call["name"])
                if fn is None:
                    result = {"error": f"Unknown tool {call['name']}"}
                else:
                    try:
                        result = fn.invoke(call["args"])
                    except Exception as e:
                        result = {"error": str(e)}
                print(f"DEBUG: Tool called: {call['name']} args={call['args']} result={result}")
                messages.append({"role": "tool", "content": str(result), "tool_call_id": call["id"]})
            response = bound_model.invoke(messages)
            messages.append(response)
            iterations += 1
            continue

        parsed = parse_malformed_tool_call(response.content)
        if parsed and parsed["name"] in tool_map:
            print(f"DEBUG: Manually parsed malformed call: {parsed}")
            try:
                result = tool_map[parsed["name"]].invoke(parsed["args"])
            except Exception as e:
                result = {"error": str(e)}
            messages[-1] = {"role": "assistant", "content": "[processing]"}
            messages.append({"role": "tool", "content": str(result), "tool_call_id": "manual_fallback"})
            response = bound_model.invoke(messages)
            messages.append(response)
            iterations += 1
            continue

        break

    agent_response = response.content or "Sorry, kuch samajh nahi aaya, dobara batayein."
    if "<function=" in agent_response:
        agent_response = "Ek second, check kar raha hoon, dobara message karein."
    return agent_response


# -------------------------------------------- Mesaage Classification Node --------------------------------------------------------------

def message_classification_node(state: State) -> State:
    message = state["usr_msg"]
    history_text = "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", [])[-4:]) or "None"
    prompt = MESSAGE_CLASSIFICATION_PROMPT_TEMPLATE.format(history_text=history_text, message=message)

    try:
        result = structured_model.invoke([SystemMessage(content=prompt), HumanMessage(content=message)])
        category, confidence = result.category, result.confidence
    except Exception as e:
        print(f"DEBUG: Structured classification failed ({e}), using fallback")
        try:
            plain = model.invoke([SystemMessage(content=prompt + "\nRespond with ONLY: booking, faq, reschedule, or other."),
                                   HumanMessage(content=message)])
            text = plain.content.strip().lower()
            category = next((c for c in ["booking", "faq", "reschedule", "other"] if c in text), "other")
            confidence = "high"
        except Exception as e2:
            print(f"DEBUG: Fallback classification also failed ({e2})")
            category, confidence = "other", "high"

    state["msg_category"] = category if confidence == "high" else "other"
    return state

# -------------------------------------------------- Router Function ---------------------------------------------------------


def router(state: State) -> Literal["faq_node", "booking_node", "reschedule_node", "other_node"]:
    category = state.get("msg_category", "other")
    return {"booking": "booking_node", "faq": "faq_node", "reschedule": "reschedule_node"}.get(category, "other_node")


# -------------------------------------------------------- Booking Node --------------------------------------------------------------

def booking_node(state: State) -> State:
    system = BOOKING_SYSTEM_PROMPT.format(
        client_phone=state["client_phone"], client_name=state.get("client_name", "Unknown"),
        today=date.today().isoformat()
    )
    user_message = state["usr_msg"]
    messages = [{"role": "system", "content": system}] + state["history"] + [{"role": "user", "content": user_message}]
    agent_response = run_tool_calling_loop(booking_model, BOOKING_TOOL_MAP, messages)
    state["response"] = agent_response
    state["history"] = state["history"] + [{"role": "user", "content": user_message}, {"role": "assistant", "content": agent_response}]
    return state


# ------------------------------------------------------- Reschedule Node -------------------------------------------------------------

def reschedule_node(state: State) -> State:
    system = RESCHEDULE_SYSTEM_PROMPT.format(
        client_phone=state["client_phone"], client_name=state.get("client_name", "Unknown"),
        today=date.today().isoformat()
    )
    user_message = state["usr_msg"]
    messages = [{"role": "system", "content": system}] + state["history"] + [{"role": "user", "content": user_message}]
    agent_response = run_tool_calling_loop(reschedule_model, RESCHEDULE_TOOL_MAP, messages)
    state["response"] = agent_response
    state["history"] = state["history"] + [{"role": "user", "content": user_message}, {"role": "assistant", "content": agent_response}]
    return state

# ----------------------------------------------- FAQ Node ---------------------------------------------------------------------------


def faq_node(state: State) -> State:
    salon_info = settings.SALON_INFO
    system = FAQ_SYSTEM_PROMPT.format(
        salon_name=salon_info["name"], address=salon_info["address"], working_hours=salon_info["working_hours"],
        phone=salon_info["phone"], policies=salon_info["policies"], client_name=state.get("client_name", "Unknown")
    )
    user_message = state["usr_msg"]
    messages = [{"role": "system", "content": system}] + state["history"] + [{"role": "user", "content": user_message}]
    agent_response = run_tool_calling_loop(faq_model, FAQ_TOOL_MAP, messages, max_iterations=3)
    state["response"] = agent_response
    state["history"] = state["history"] + [{"role": "user", "content": user_message}, {"role": "assistant", "content": agent_response}]
    return state

# ---------------------------------------------- Esclate Node -------------------------------------------------------------

def other_node(state: State) -> State:
    user_message = state["usr_msg"]
    try:
        check = model.invoke(ESCALATE_CLASSIFY_PROMPT.format(message=user_message))
        needs_staff = "needs_staff" in check.content.lower()
    except Exception as e:
        print(f"DEBUG: Escalate-classify failed ({e}), defaulting to needs_staff")
        needs_staff = True

    if not needs_staff:
        try:
            greeting_prompt = GREETING_RESPONSE_PROMPT.format(client_name=state.get("client_name", "Unknown"))
            response = model.invoke([{"role": "system", "content": greeting_prompt}, {"role": "user", "content": user_message}])
            agent_response = response.content
        except Exception as e:
            print(f"DEBUG: Greeting response failed ({e})")
            agent_response = "Wa alaikum salam! Main aapki kya madad kar sakta hoon?"
        state["escalate"] = False
    else:
        agent_response = "Maazrat chahtay hain, is baare mein main madad nahi kar sakta. Hamara staff jald hi aap se rabta karega."
        state["escalate"] = True
        try:
            notify_staff(state["client_phone"], user_message)
        except Exception as e:
            print(f"DEBUG: notify_staff failed ({e})")

    state["response"] = agent_response
    state["history"] = state["history"] + [{"role": "user", "content": user_message}, {"role": "assistant", "content": agent_response}]
    return state


# ------------------------------------------ Global safety wrapper --------------------------------------------------------

def safe_node_wrapper(node_func):
    def wrapped(state: State) -> State:
        try:
            return node_func(state)
        except Exception as e:
            print(f"DEBUG: Node {node_func.__name__} crashed: {e}")
            state["response"] = "Maazrat, kuch technical masla hua hai. Thodi der baad dobara koshish karein."
            state["escalate"] = True
            state["history"] = state.get("history", []) + [
                {"role": "user", "content": state.get("usr_msg", "")},
                {"role": "assistant", "content": state["response"]}
            ]
            return state
    return wrapped


# ----------------------------------------------------- Graph Creation -----------------------------------------------------------------

workflow = StateGraph(State)
workflow.add_node("message_classification_node", safe_node_wrapper(message_classification_node))
workflow.add_node("booking_node", safe_node_wrapper(booking_node))
workflow.add_node("reschedule_node", safe_node_wrapper(reschedule_node))
workflow.add_node("faq_node", safe_node_wrapper(faq_node))
workflow.add_node("other_node", safe_node_wrapper(other_node))

workflow.add_edge(START, "message_classification_node")
workflow.add_conditional_edges("message_classification_node", router)
workflow.add_edge("booking_node", END)
workflow.add_edge("reschedule_node", END)
workflow.add_edge("faq_node", END)
workflow.add_edge("other_node", END)

graph = workflow.compile()
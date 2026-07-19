from django.shortcuts import render
import json
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from twilio.twiml.messaging_response import MessagingResponse
from salon.models import Client, Conversation
from agent.graph import graph


@csrf_exempt
def whatsapp_webhook(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=405)

    # ---- Incoming message parse karo (Twilio form-data bhejta hai, JSON nahi) ----
    try:
        from_number = request.POST.get("From", "").replace("whatsapp:", "")
        text = request.POST.get("Body", "")
        contact_name = request.POST.get("ProfileName", "Unknown")
    except Exception as e:
        print(f"DEBUG: Malformed Twilio payload: {e}")
        resp = MessagingResponse()
        return HttpResponse(str(resp), content_type="text/xml")

    if not from_number or not text:
        resp = MessagingResponse()
        return HttpResponse(str(resp), content_type="text/xml")

    # ---- Client + Conversation load/create ----
    try:
        client, _ = Client.objects.get_or_create(phone=from_number, defaults={"name": contact_name})
        conv, _ = Conversation.objects.get_or_create(
            client=client, channel="whatsapp", defaults={"transcript": []}
        )
        history = conv.transcript or []
    except Exception as e:
        print(f"DEBUG: DB error loading client/conversation: {e}")
        history = []
        conv = None

    # ---- Agent invoke karo ----
    try:
        result = graph.invoke({
            "usr_msg": text,
            "msg_category": "",
            "client_name": contact_name,
            "client_phone": from_number,
            "response": None,
            "booking_context": {},
            "escalate": False,
            "history": history
        })
        reply_text = result.get("response") or "Maazrat, kuch masla hua. Dobara koshish karein."
    except Exception as e:
        print(f"DEBUG: Graph invoke fully crashed: {e}")
        reply_text = "Maazrat, is waqt system available nahi hai. Hamara staff jald rabta karega."
        result = {"history": history, "escalate": True}

    # ---- History save karo ----
    if conv is not None:
        try:
            conv.transcript = result.get("history", history)
            conv.resolved = not result.get("escalate", False)
            conv.save()
        except Exception as e:
            print(f"DEBUG: Failed to save conversation: {e}")

    # ---- TwiML response return karo — Twilio khud isay WhatsApp message mein bhej deta hai ----
    resp = MessagingResponse()
    resp.message(reply_text)
    return HttpResponse(str(resp), content_type="text/xml")


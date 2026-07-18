from typing import Literal,Optional,List,TypedDict,Union

class State(TypedDict):
    usr_msg: str
    msg_category: str
    
    ## State for booking
    client_name : str
    client_phone : str
    response : Optional[str]
    booking_context : dict
    escalate : bool
    history: List[dict] # for short term memory taa k LLM context bhule nhi 
    
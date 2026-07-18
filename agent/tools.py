from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from pydantic import BaseModel,Field
from typing import Literal,Optional,List,TypedDict,Union
from langgraph.graph import START,END,StateGraph
import operator
from langchain_core.tools import tool
from salon.models import Staff, Service, Client, Appointment
from datetime import timedelta
from django.utils import timezone
import datetime
from datetime import datetime, date, timedelta

from django.conf import settings
import re
import json


# -------------------------------------------------- Helper Functions ------------------------------------------------------------

# 1. Parse start time
def parse_start_time(start_time_str: str):
    """Accepts both naive ('2026-07-19T09:00') aur aware 
    ('2026-07-19T09:00:00+00:00') ISO strings, hamesha aware datetime return karta hai."""
    dt = datetime.fromisoformat(start_time_str)
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt)
    return dt

# 2. Generate Slots
def generate_slots(start_time, end_time, duration_minutes, date_str):
    """Working hours ke andar duration ke hisaab se possible slots banata hai."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    current = timezone.make_aware(datetime.combine(date_obj, start_time))
    end_dt = timezone.make_aware(datetime.combine(date_obj, end_time))
    slots = []
    while current + timedelta(minutes=duration_minutes) <= end_dt:
        slots.append(current)
        current += timedelta(minutes=duration_minutes)
    return slots

# 3. Find Alternatives
def _find_alternatives(requested_staff, service, date_str, exclude_time):
    """Same staff ke doosre free slots, aur doosre staff members ke slots dono dhoondo."""
    alternatives = []

    # 1) Pehle same staff ke doosre free slots dekho (same din)
    same_staff_slots = _get_free_slots_for_staff(requested_staff, service, date_str, exclude_time)
    if same_staff_slots:
        alternatives.append({"staff": requested_staff.name, "slots": same_staff_slots[:3]})

    # 2) Agar same staff ke paas kam/koi slot nahi, doosre staff bhi suggest karo
    other_staff = Staff.objects.exclude(id=requested_staff.id)
    for staff in other_staff:
        slots = _get_free_slots_for_staff(staff, service, date_str, exclude_time)
        if slots:
            alternatives.append({"staff": staff.name, "slots": slots[:3]})
        if len(alternatives) >= 4:   # zyada options se overwhelm na karo
            break

    return alternatives

# 4. get free slots for staff
def _get_free_slots_for_staff(staff, service, date_str, exclude_time=None):
    booked_times = set(
        Appointment.objects.filter(
            staff=staff, start_time__date=date_str,
            status__in=["pending", "confirmed"]
        ).values_list("start_time", flat=True)
    )
    possible_slots = generate_slots(
        staff.working_hours_start, staff.working_hours_end,
        service.duration_minutes, date_str
    )
    free = [s for s in possible_slots if s not in booked_times]
    return [str(s.time()) for s in free]

# 4. Notify Staff
def notify_staff(client_phone: str, message: str):
    """Owner ko WhatsApp pe alert bhejta hai jab agent escalate kare."""
    owner_number = settings.SALON_WHATSAPP_NUMBER   # ye actually staff ka apna number hona chahiye
    alert_text = (
        f"⚠️ Customer ({client_phone}) ko madad chahiye:\n"
        f"\"{message}\"\n"
        f"Please WhatsApp par check karein."
    )
    # abhi ke liye print — real WhatsApp integration Phase 3 mein hogi
    print(f"STAFF ALERT: {alert_text}")

# 5. Find Services
def find_service(service_name: str):
    """Spaces/case ignore kar ke flexible matching."""
    service = Service.objects.filter(name__icontains=service_name).first()
    if service:
        return service
    normalized = service_name.lower().replace(" ", "")
    for s in Service.objects.all():
        if normalized in s.name.lower().replace(" ", ""):
            return s
    return None

# 6. Malinformed tools
def looks_like_malformed_tool_call(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"<function=", text))


def parse_malformed_tool_call(text: str):
    match = re.search(r"<function=(\w+)>(\{.*?\})</function>", text)
    if not match:
        return None
    try:
        return {"name": match.group(1), "args": json.loads(match.group(2))}
    except json.JSONDecodeError:
        return None



# --------------------------------------------------------- Tool Schema -----------------------------------------------------------------------
class CheckAvailabilityInput(BaseModel):
    service_name: str = Field(description="Name of the service, e.g. 'facial', 'haircut'")
    preferred_date: str = Field(description="Date in YYYY-MM-DD format")
    

class CreateAppointmentInput(BaseModel):
    client_phone: str = Field(description="Customer's WhatsApp phone number")
    client_name: str = Field(description="Customer's name")
    staff_name: str = Field(description="Name of the staff member")
    service_name: str = Field(description="Name of the service")
    start_time: str = Field(description="Appointment start time in YYYY-MM-DDTHH:MM format")
    
    
class FindAppointmentInput(BaseModel):
    client_phone: str = Field(description="Customer's WhatsApp phone number")
    

class UpdateAppointmentInput(BaseModel):
    appointment_id: Union[int, str] = Field(description="ID of the appointment to reschedule")
    new_start_time: str = Field(description="New start time in YYYY-MM-DDTHH:MM format")


class CancelAppointmentInput(BaseModel):
    appointment_id: Union[int, str] = Field(description="ID of the appointment to cancel")
    

# ------------------------------------------------------------------ Tools ----------------------------------------------------------------

# 1. Booking Tools
@tool(args_schema=CheckAvailabilityInput)
def check_availability(service_name: str, preferred_date: str) -> dict:
    """Check available appointment slots for a service on a given date."""
    service = find_service(service_name)
    if not service:
        return {"error": f"'{service_name}' service not found"}

    available = []
    for staff in Staff.objects.all():
        booked_times = set(
            Appointment.objects.filter(
                staff=staff, start_time__date=preferred_date,
                status__in=["pending", "confirmed"]
            ).values_list("start_time", flat=True)
        )
        possible_slots = generate_slots(
            staff.working_hours_start, staff.working_hours_end,
            service.duration_minutes, preferred_date
        )
        free_slots = [s for s in possible_slots if s not in booked_times]
        if free_slots:
            available.append({"staff": staff.name, "slots": [str(s.time()) for s in free_slots[:3]]})

    if not available:
        return {"service": service.name, "date": preferred_date, "availability": [], "message": "No slots free that day"}
    return {"service": service.name, "date": preferred_date, "availability": available}


@tool(args_schema=CreateAppointmentInput)
def create_appointment(client_phone: str, client_name: str, staff_name: str,
                        service_name: str, start_time: str) -> dict:
    """Book the appointment. ONLY call after explicit confirmation."""
    client, _ = Client.objects.get_or_create(phone=client_phone, defaults={"name": client_name})
    staff = Staff.objects.filter(name__icontains=staff_name).first()
    service = find_service(service_name)
    if not staff or not service:
        return {"error": "Staff or service not found"}

    start = parse_start_time(start_time)
    end = start + timedelta(minutes=service.duration_minutes)
    date_str = start.date().isoformat()

    clash = Appointment.objects.filter(
        staff=staff, start_time__lt=end, end_time__gt=start,
        status__in=["pending", "confirmed"]
    ).exists()
    if clash:
        alternatives = _find_alternatives(staff, service, date_str, exclude_time=start)
        return {"success": False, "error": f"{staff.name} is not available at {start.time()} on {date_str}",
                "alternatives": alternatives}

    appt = Appointment.objects.create(
        client=client, staff=staff, service=service,
        start_time=start, end_time=end, status="confirmed", booked_by_agent=True
    )
    return {"success": True, "appointment_id": appt.id, "time": start_time, "staff": staff.name}


@tool
def get_services_list() -> dict:
    """Get the current list of services with prices and durations."""
    services = Service.objects.all()
    return {"services": [
        {"name": s.name, "price": str(s.price) if s.price else "Contact for price",
         "duration_minutes": s.duration_minutes}
        for s in services
    ]}

# 2. Reschedule Tools
@tool(args_schema=FindAppointmentInput)
def find_appointment(client_phone: str) -> dict:
    """Find the customer's upcoming appointments."""
    client = Client.objects.filter(phone=client_phone).first()
    if not client:
        return {"found": False, "message": "No client record found for this number"}
    upcoming = Appointment.objects.filter(
        client=client, status__in=["pending", "confirmed"], start_time__gte=timezone.now()
    ).order_by("start_time")
    if not upcoming.exists():
        return {"found": False, "message": "No upcoming appointments found"}
    return {"found": True, "appointments": [
        {"appointment_id": a.id, "service": a.service.name if a.service else "N/A",
         "staff": a.staff.name if a.staff else "N/A", "time": a.start_time.isoformat()}
        for a in upcoming
    ]}


@tool(args_schema=UpdateAppointmentInput)
def update_appointment(appointment_id, new_start_time: str) -> dict:
    """Reschedule an existing appointment."""
    appointment_id = int(appointment_id)
    appt = Appointment.objects.filter(id=appointment_id).first()
    if not appt:
        return {"success": False, "error": "Appointment not found"}

    new_start = parse_start_time(new_start_time)
    new_end = new_start + timedelta(minutes=appt.service.duration_minutes)
    date_str = new_start.date().isoformat()

    clash = Appointment.objects.filter(
        staff=appt.staff, start_time__lt=new_end, end_time__gt=new_start,
        status__in=["pending", "confirmed"]
    ).exclude(id=appt.id).exists()
    if clash:
        alternatives = _find_alternatives(appt.staff, appt.service, date_str, exclude_time=new_start)
        return {"success": False, "error": f"{appt.staff.name} is not available at {new_start.time()} on {date_str}",
                "alternatives": alternatives}

    appt.start_time = new_start
    appt.end_time = new_end
    appt.save()
    return {"success": True, "appointment_id": appt.id, "new_time": new_start_time}


@tool(args_schema=CancelAppointmentInput)
def cancel_appointment(appointment_id) -> dict:
    """Cancel an existing appointment."""
    appointment_id = int(appointment_id)
    appt = Appointment.objects.filter(id=appointment_id).first()
    if not appt:
        return {"success": False, "error": "Appointment not found"}
    appt.status = "cancelled"
    appt.save()
    return {"success": True, "appointment_id": appt.id, "message": "Appointment cancelled"}


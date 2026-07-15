from django.db import models

# ----------------------------------------- STAFF -------------------------------
class Staff(models.Model):
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100, blank=True)   # "Hair stylist", "Beautician"
    working_hours_start = models.TimeField(default="09:00")
    working_hours_end = models.TimeField(default="18:00")
    off_day = models.CharField(max_length=20, blank=True, help_text="e.g. Sunday")

# --------------------------------------- Services -------------------------------------
class Service(models.Model):
    name = models.CharField(max_length=150)          # "Facial", "Haircut", "Manicure"
    duration_minutes = models.PositiveIntegerField(default=30)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

# ------------------------------------- Client ----------------------------------------
class Client(models.Model):
    name = models.CharField(max_length=150)
    phone = models.CharField(max_length=20, unique=True, db_index=True)
    notes = models.TextField(blank=True)               # history, preferences
    created_at = models.DateTimeField(auto_now_add=True)

# ------------------------------------- Appointments ------------------------------------
class Appointment(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("confirmed", "Confirmed"),
        ("cancelled", "Cancelled"),
        ("completed", "Completed"),
        ("no_show", "No-show"),
    ]

    client = models.ForeignKey(Client, on_delete=models.CASCADE, related_name="appointments")
    staff = models.ForeignKey(Staff, on_delete=models.SET_NULL, null=True)
    service = models.ForeignKey(Service, on_delete=models.SET_NULL, null=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    booked_by_agent = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["staff", "start_time"])]

# ------------------------------------- Conversation -----------------------------------

class Conversation(models.Model):
    client = models.ForeignKey(Client, on_delete=models.CASCADE, null=True, related_name="conversations")
    channel = models.CharField(max_length=20, default="whatsapp")
    transcript = models.JSONField(default=list)
    resolved = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

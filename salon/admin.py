from django.contrib import admin
from .models import Staff, Service, Client, Appointment, Conversation

# Register your models here.



admin.site.register(Staff)
admin.site.register(Service)
admin.site.register(Client)
admin.site.register(Appointment)
admin.site.register(Conversation)

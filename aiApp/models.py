import mongoengine
from django.db import models


# Create your models here.
class CncTool(mongoengine.Document):
    channel_name = mongoengine.StringField(max_length=20)
    value = mongoengine.FloatField()
    timestamp = mongoengine.DateTimeField()
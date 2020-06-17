from django import template
from threading import Thread
from time import sleep

a = 1
register = template.Library()
@register.filter
def add_image():
    sleep(0.1)
    s = "a"+".PNG"
    a = a + 1 
    return s

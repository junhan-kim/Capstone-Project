from django.shortcuts import render
from django.views.generic import TemplateView
# Create your views here.


class CctvView_c(TemplateView):
    
    template_name = 'home.html'

class CctvView_f(TemplateView):
    
    template_name = 'fire.html'

    
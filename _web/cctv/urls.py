from django.urls import path
from . import views
app_name = 'cctv'
urlpatterns=[
    path('cctv/crash', views.CctvView_c.as_view() ,name ='home'),
    path('cctv/fire',views.CctvView_f.as_view(),name = 'fire')
,]
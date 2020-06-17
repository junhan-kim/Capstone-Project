from django.shortcuts import render
from django.views.generic.list import BaseListView
from django.views.generic.edit import BaseDeleteView
from django.views.generic.edit import BaseCreateView
from django.http import JsonResponse
from todo.models import Todo
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
from django.forms.models import model_to_dict


# Create your views here.
class ApiTodoLV(BaseListView):
    model = Todo

    # def get(self,request,*args,**kwargs):
    #     tmpList =[
    #             {'id' :1, 'name': '김석훈', 'todo': 'Django 와 Vue.js 연동 프로그램을 만들고 있습니다.'},
    #             {'id' :2, 'name': '홍길동', 'todo': '이름을 안쓰면 홍길동으로 나와요...'},
    #             {'id' :3, 'name': '이순신', 'todo': '신에게는 아직 열두 척의 배가 있사옵니다.'},
    #             {'id' :, 'name': '성춘향', 'todo': '그네 타기'},            
    #     ]
    #     return JsonResponse(data=tmpList, safe=False)

    def render_to_response(self, context, **response_kwargs):
        todolist = list(context['object_list'].values())
        return  JsonResponse(data=todolist ,safe =False)
    

@method_decorator(csrf_exempt,name='dispatch')
class ApiTodoDelV(BaseDeleteView):
    model = Todo

    def delete(self,request,*args,**kwargs):
        self.object =self.get_object()
        self.object.delete()
        return JsonResponse(data={},status=204)

@method_decorator(csrf_exempt,name='dispatch')
class ApiTodoCV(BaseCreateView):
    model = Todo
    fields = '__all__'

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['data'] =json.loads(self.request.body)
        return kwargs


    def form_valid(self,form):
        self.object =form.save()
        newTodo = model_to_dict(self.object)
        return JsonResponse(data=newTodo,status = 201)

    def form_invalid(self,form):
        return JsonResponse(data=form.errors,status =400)







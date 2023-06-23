from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.utils import timezone
from django.http import JsonResponse
from django.template.loader import render_to_string
from .forms import DataForm
from .my_module import cal
import time
import os

Cal=cal.Cal()
Winds=['東','南','西','北']
default_p={ '3nan':[[33.06,34.70,32.23],[33.22,35.22,31.56],[33.72,30.08,36.21]],
            '4ton':[[24.88,24.93,25.58,24.60],[24.82,25.13,25.95,24.10],[25.02,25.22,26.03,23.73],[25.28,24.71,22.44,27.57]],
            '4nan':[[24.75,25.08,25.36,24.82],[24.80,25.06,25.56,24.57],[25.04,25.07,25.70,24.18],[25.41,24.79,23.38,26.43]]}



def index(request):
    form = DataForm()
    return render(request, 'mj_app/index.html', {})


def make_result(request):
    num_of_people=int(request.POST.get('num_of_people'))
    num_of_kyoku=request.POST.get('num_of_kyoku')
    bakaze = int(request.POST.get('bakaze'))
    kyoku = int(request.POST.get('kyoku'))
    score0 = int(request.POST.get('score0'))
    score1 = int(request.POST.get('score1'))
    score2 = int(request.POST.get('score2'))
    score3 = int(request.POST.get('score3'))

    data=[score0,score1,score2,score3]
    data=data[:num_of_people]
    kyoku_t=kyoku-1+num_of_people*bakaze
    if kyoku_t==0 and score0==score1==score2 and (num_of_people==3 or (num_of_people==4 and score2==score3)):
        result=default_p[f'{num_of_people}{num_of_kyoku}']
    else:result=Cal.predict(num_of_people,num_of_kyoku,kyoku_t,data)
    #result=[0]

    info=['']
    for i in range(num_of_people):
        info.append(f'{num_of_people-i}位率')

    players=[]
    for i in range(num_of_people):
        player={}
        player['name']=f'Player {i+1} : {Winds[(i-kyoku+1)%num_of_people]}'
        p=[0]*num_of_people
        for j in range(num_of_people):
            p[j]=f'{round(result[i][j],2)}%'
        player['p']=p
        players.append(player)

    context={}
    context['info']=info
    context['players']=players
    return render_to_string('mj_app/result.html',context)

def execute(request):
    json={}
    json["result"]= make_result(request)
    return JsonResponse(json)
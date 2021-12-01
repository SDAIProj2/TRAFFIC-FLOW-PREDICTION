from django.shortcuts import render
from django.http import HttpResponse

from django.shortcuts import render
from .forms import InputForm


# generate random integer values
from random import seed
from random import randint
# seed random number generator
seed(1)
# generate some integers


def index(request):
    context = {}
    context['form'] = InputForm()
    return render(request, "home.html", context)
def index2(request):
    value = randint(0, 15)
    context = {}
    if value < 5:
        return render(request, "home2.html", context)
    elif value > 5 and value < 10:
        return render(request, "home3.html", context)
    else:
        return render(request, "home4.html", context)



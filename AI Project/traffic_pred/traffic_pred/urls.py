
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('traffic_app/', include('traffic_app.urls')),
    path('admin/', admin.site.urls),
]

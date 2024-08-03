"""
URL configuration for AIHumanResource project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [

    path('', views.home, name='home'),
    path('new-job/', views.NewJob.as_view(), name='new-job'),
    path('edit-job/<int:job_id>/', views.EditJob.as_view(), name='edit-job'),
    path('careers/', views.careers, name='careers'),
    path('job-detail/<int:job_id>/', views.job_detail_page, name='job-detail'),
    path('app-job/<int:job_id>/', views.ApplyJob.as_view(), name='apply-job'),
    path('shortlist/<int:job_id>/', views.shortlist, name='shortlist'),
    path('shortlist-candidates/<int:job_id>/', views.shortlist_candidates, name='shortlist-candidates')


]

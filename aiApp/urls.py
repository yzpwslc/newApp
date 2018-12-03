#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:21:20 2018

@author: yzp1011
"""

from django.urls import path
from . import views


urlpatterns = [
        path('', views.index, name='index'),
        ]
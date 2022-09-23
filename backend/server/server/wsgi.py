"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
from apps.ml.income_classifier.mlp import MLP  # import Multi Layer Perceptron ML algorithm

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

try:
    mlp = MLP()
except Exception as e:
    print("Exception while loading the algorithms, ", str(e))

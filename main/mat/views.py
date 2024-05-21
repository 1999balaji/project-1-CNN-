import pickle

from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
import numpy as np
import cv2
import sklearn.neural_network
class first(View):

    def get(self, request):
        return render(request, 'abcd.html')

    def post(self, request):

        emt=np.zeros((1,2704))
        for f in request.FILES.getlist("img"):
            img = f.read()
            image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
            gr=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


            haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            faces_rect = haar.detectMultiScale(gr,scaleFactor=1.1,minNeighbors=9,flags=cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w,h) in faces_rect:
                gr= gr[y:y + h,x:x + w]


            res=cv2.resize(gr,(52,52))

            rs=res.reshape(1,2704)
            emt=np.vstack([emt,rs])
        newar = np.delete(emt, 0,0)


        mdl=sklearn.neural_network.MLPClassifier((100, 50))
        mdl.fit(newar,[0,0,0,0,0,1,1,1,1,1,])

        with open ('mod.plk','wb')as f:
            pickle.dump(mdl,f)

        return HttpResponse('see you and buy ')


class second(View):

    def get(self, request):
        return render(request, 'check.html')

    def post(self, request):

        emt=np.zeros((1,2704))
        for f in request.FILES.getlist("img"):
            img = f.read()
            image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
            gr=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


            haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            faces_rect = haar.detectMultiScale(gr,scaleFactor=1.1,minNeighbors=9,flags=cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w,h) in faces_rect:
                gr= gr[y:y + h,x:x + w]


            res=cv2.resize(gr,(52,52))

            rs=res.reshape(1,2704)

        with open ('mod.plk','rb')as f:
            pkmod=pickle.load(f)
        prd=pkmod.predict(rs)
        print(prd)

        return HttpResponse('done')

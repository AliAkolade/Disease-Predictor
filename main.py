import pickle
import threading
import time

import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.label import Label
from kivy.uix.popup import Popup


class WindowManager(ScreenManager):
    pass


class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.get_screen)

    def get_screen(self, a):
        threading.Thread(target=self.change_screen).start()

    @mainthread
    def change_screen(self):
        time.sleep(0.5)
        self.manager.current = "Select"


class SelectScreen(Screen):
    def all_page(self):
        self.parent.current = 'Predict'


def basic_models(path, X_test, f_scale):
    X_test = [X_test]
    loaded_model = pickle.load(open(path, 'rb'))
    if f_scale:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test = sc.fit_transform(X_test)
    y_pred = loaded_model.predict(X_test)
    return y_pred


class PredictScreen(Screen):
    diabetes_risk = 0
    cvd_risk = 0
    hyper_risk = 0
    stress_risk = 0
    cvs = 0

    def diabetes(self):
        age = float(self.ids.input_age.text)
        preg = float(self.ids.input_preg.text)
        glu = float(self.ids.input_glu.text)
        bldsgr = float(self.ids.input_bldsgr.text)
        sknthck = float(self.ids.input_sknthck.text)
        insln = float(self.ids.input_insln.text)
        bmi = float(self.ids.input_bmi.text)
        dpf = float(self.ids.input_dpf.text)

        X_test = [preg, glu, bldsgr, sknthck, insln, bmi, dpf, age]

        path = '1 Diabetes/Models/'
        dia_1 = basic_models(path + 'K Means Clustering Diabetes.sav', X_test, False)[0]
        dia_2 = basic_models(path + 'Naive Bayes Diabetes.sav', X_test, False)[0]
        if dia_1 == dia_2:
            if dia_1 == 1:
                print("High Chance of Diabetes")
                dia = 85
            else:
                print("Low Chance of Diabetes")
                dia = 20
        else:
            print("Medium Chance of Diabetes")
            dia = 50
        return dia

    def cvd(self):
        age = float(self.ids.input_age.text)
        gender = float(self.ids.input_gen.text)
        height = float(self.ids.input_h.text)
        weight = float(self.ids.input_w.text)
        chol = float(self.ids.input_chl.text)
        s_bp = float(self.ids.input_sbp.text)
        d_bp = float(self.ids.input_dbp.text)
        bmi = float(self.ids.input_bmi.text)
        smk = float(self.ids.input_smk.text)
        glucose = float(self.ids.input_glu.text)
        glucose = 1 if glucose <= 100 else (2 if glucose <= 200 else 3)

        X_test = [age, gender, height, weight, bmi, s_bp, d_bp, chol, glucose, smk]

        path = '2 CVD/Models/'
        mdl_1 = basic_models(path + 'Logistic Regression 4.sav', X_test, False)[0]
        mdl_2 = basic_models(path + 'Decision Tree Classifier.sav', X_test, False)[0]
        mdl_3 = basic_models(path + 'Logistic Regression 3.sav', X_test, False)[0]
        mdl_4 = basic_models(path + 'Random Forest Classifier.sav', X_test, False)[0]
        if mdl_1 == mdl_2 == mdl_3 == mdl_4:
            if mdl_1 == 1:
                print("High Chance of CVD")
                mdl = 90
            else:
                print("Low Chance of CVD")
                mdl = 30
        else:
            print("Medium Chance of CVD")
            mdl = 60
        return mdl

    def hyper(self):
        wc = float(self.ids.input_wc.text)
        gender = float(self.ids.input_gen.text)
        s_bp = float(self.ids.input_sbp.text)
        d_bp = float(self.ids.input_dbp.text)
        valid_X = [gender], [wc], [s_bp], [d_bp]
        valid_X = np.array(valid_X)
        valid_X = valid_X.reshape(1, 4)
        mdl = basic_models('3 Hypertension/Models/Hypertension MLP.sav', valid_X, False)
        if mdl < 0.45:
            print("Low Chance of Hypertension")
        if 0.65 > mdl >= 0.45:
            print("Medium Chance of Hypertension")
        if 0.65 <= mdl < 0.85:
            print("High Chance of Hypertension")
        if mdl >= 0.85:
            print("Very High Chance of Hypertension")
        return int(mdl[0][0] * 100)

    def stress(self):
        path = '4 Stress/Models/Stress GNB NonNorm.sav'
        ecg = float(self.ids.input_ecg.text)
        emg = float(self.ids.input_emg.text)
        f_gsr = float(self.ids.input_f_gsr.text)
        h_gsr = float(self.ids.input_h_gsr.text)
        hr = float(self.ids.input_hr.text)
        resp = float(self.ids.input_resp.text)
        X_test = [ecg, emg, f_gsr, h_gsr, hr, resp]
        mdl = basic_models(path, X_test, False)[0]
        if mdl == 1:
            mdl = 74
            print("High Chance of Stress")
        if mdl == 0:
            mdl = 24
            print("Low Chance of Stress")
        return mdl

    def cvs(self):
        comp = True if float(self.ids.input_comp.text) == 1 else False
        hrs = float(self.ids.input_hrs.text)
        ciu = True if float(self.ids.input_cui.text) == 1 else False
        exp = True if float(self.ids.input_exp.text) == 1 else False
        if comp and hrs >= 4 and ciu and exp:
            mdl = 80
            print("High Chance of Computer Vision Syndrome")
        elif comp and hrs >= 4 and exp:
            mdl = 63
            print("Medium Chance of Computer Vision Syndrome")
        elif not comp:
            mdl = 10
            print("Low Chance of Computer Vision Syndrome")
        elif comp and exp:
            mdl = 59
            print("Medium Chance of Computer Vision Syndrome")
        return mdl



    def calculate(self):
        r_list = {'Diabetes': self.diabetes_risk, 'Cardiovascular Diseases': self.cvd_risk,
                  'Hypertension': self.hyper_risk, 'Stress': self.stress_risk, 'Computer Vision Syndrome': self.cvs}
        values = list(r_list.values())
        values.sort(reverse=True)
        names = ["", "", "", "", ""]
        for i in range(0, 5):
            for key, value in r_list.items():
                if int(values[i]) == int(value):
                    names[i] = key
        result = "Highest Risk Disease = "+str(names[0])+"\n\nInfo - \n"+str(names[0])+" - "+str(values[0])+"%\n"+str(names[1] +" - "+\
                 str(values[1])+"%\n"+str(names[2])+" - "+str(values[2])+"%\n"+str(names[3])+" - "+str(values[3])+"%\n" + names[4])+" - "+\
                 str(values[4])+"%\n"
        return str(result)

    def showResult(self):
        result = Label(text=self.calculate())
        popupWindow = Popup(title="Result", title_align='center', content=result, size_hint=(0.7, 0.7))
        popupWindow.open()

    def predict(self):
        self.diabetes_risk = self.diabetes()
        print(self.diabetes_risk)
        self.cvd_risk = self.cvd()
        print(self.cvd_risk)
        self.hyper_risk = self.hyper()
        print(self.hyper_risk)
        self.stress_risk = self.stress()
        print(self.stress_risk)
        self.cvs = self.cvs()
        print(self.cvs)
        self.showResult()


class SingleScreen(Screen):
    pass


class Gui(App):
    def build(self):
        return WindowManager()


if __name__ == '__main__':
    Gui().run()

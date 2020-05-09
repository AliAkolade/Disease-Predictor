import pickle
import threading
import time

import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.uix.screenmanager import Screen, ScreenManager


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
                dia = 100
            else:
                print("Low Chance of Diabetes")
                dia = 0
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
                mdl = 100
            else:
                print("Low Chance of CVD")
                mdl = 0
        else:
            print("Medium Chance of CVD")
            mdl = 50
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
        ecg = float(self.ids.input_wc.text)
        emg = float(self.ids.input_gen.text)
        f_gsr = float(self.ids.input_sbp.text)
        h_gsr = float(self.ids.input_dbp.text)
        hr = float(self.ids.input_sbp.text)
        resp = float(self.ids.input_dbp.text)
        X_test = [ecg, emg, f_gsr, h_gsr, hr, resp]
        mdl = basic_models(path, X_test, False)[0]
        if mdl == 1:
            print("High Chance of Stress")
        if mdl == 0:
            print("Low Chance of Stress")
        return mdl

    def predict(self):
        diabetes_risk = self.diabetes()
        print(diabetes_risk)
        cvd_risk = self.cvd()
        print(cvd_risk)
        hyper_risk = self.hyper()
        print(hyper_risk)
        stress_risk = self.stress()
        print(stress_risk)


class SingleScreen(Screen):
    pass


class Gui(App):
    def build(self):
        return WindowManager()


if __name__ == '__main__':
    Gui().run()

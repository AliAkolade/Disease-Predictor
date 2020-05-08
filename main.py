import pickle
import threading
import time

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen, ScreenManager


class WindowManager(ScreenManager):
    pass


class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.get_screen)

    def get_screen(self, dt):
        threading.Thread(target=self.change_screen).start()

    @mainthread
    def change_screen(self):
        time.sleep(1)
        self.manager.current = "Select"


class SelectScreen(Screen):
    def all_page(self):
        self.parent.current = 'Predict'


def basic_models(path, X_test, f_scal):
    X_test = [X_test]
    loaded_model = pickle.load(open(path, 'rb'))
    if f_scal:
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
        dia_1 = basic_models(path+'K Means Clustering Diabetes.sav', X_test, False)[0]
        dia_2 = basic_models(path+'Naive Bayes Diabetes.sav', X_test, False)[0]
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

    def predict(self):
        diabetes_risk = self.diabetes()
        print(diabetes_risk)


class SingleScreen(Screen):
    pass


class Gui(App):
    def build(self):
        return WindowManager()


if __name__ == '__main__':
    Gui().run()

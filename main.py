import threading
import time

from kivy.app import App
from kivy.clock import Clock, mainthread

from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder


class WindowManager(ScreenManager):
    pass


class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.get_screen)

    def get_screen(self, dt):
        threading.Thread(target=self.change_screen).start()

    @mainthread
    def change_screen(self):
        time.sleep(3)
        self.manager.current = "Select"


class SelectScreen(Screen):
    def all_page(self):
        self.parent.current = 'Predict'


class PredictScreen(Screen):
    pass


class Gui(App):
    def build(self):
        return WindowManager()


if __name__ == '__main__':
    Gui().run()

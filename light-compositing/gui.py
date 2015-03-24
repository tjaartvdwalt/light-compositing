import cv2
from gi.repository import Gtk, Gdk, GdkPixbuf


class MainWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Grid Example")
        self.grid = Gtk.Grid()
        self.add(self.grid)        
        self.img_gtk = None

        self.displayImage(img)

        # grid = Gtk.Grid()
        # self.add(grid)
        # cv2.createTrackbar
        button1 = Gtk.Button(label="Button 1")
        # button2 = Gtk.Button(label="Button 2")
        # button3 = Gtk.Button(label="Button 3")
        # button4 = Gtk.Button(label="Button 4")
        # button5 = Gtk.Button(label="Button 5")
        # button6 = Gtk.Button(label="Button 6")

        self.grid.add(button1)
        # grid.attach(button2, 1, 0, 2, 1)
        # grid.attach_next_to(button3, button1, Gtk.PositionType.BOTTOM, 1, 2)
        # grid.attach_next_to(button4, button3, Gtk.PositionType.RIGHT, 2, 1)
        # grid.attach(button5, 1, 2, 1, 1)
        # grid.attach_next_to(button6, button5, Gtk.PositionType.RIGHT, 1, 1)

    def displayImage(self, img):
        if self.img_gtk is None:
            self.img_gtk = Gtk.Image()
            self.grid.add(self.img_gtk)

        rgb_img = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)
        img_pixbuf = GdkPixbuf.Pixbuf.new_from_data(rgb_img.data,
                                                    GdkPixbuf.Colorspace.RGB,
                                                    False,
                                                    8,
                                                    rgb_img.shape[1],
                                                    rgb_img.shape[0],
                                                    rgb_img.shape[1] * 3,
                                                    None)
        print rgb_img.data
        image = Gtk.Image.new_from_pixbuf(img_pixbuf)
        self.grid.add(image)

img_name = "test_data/000.png"
img = cv2.imread(img_name)
win = MainWindow()
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()

import wx

class MainFrame(wx.Frame):
    def __init__(self):
        super(MainFrame, self).__init__(None, title = 'Setup', size = (1200, 650))
        
        self.Centre()

if __name__ == '__main__':
    app = wx.App()
    
    frame = MainFrame()
    frame.Show()

    app.MainLoop()
import wx
from utils import settings
from menu import menubar
class MainFrame(wx.Frame):
    def __init__(self):
        super(MainFrame, self).__init__(None, title = 'Setup', size = (1200, 650))
        
        self.SetMenuBar(menubar.MenuBar())
        self.Bind(wx.EVT_MENU, self.OnQuit, id=settings.MenuIDS['FILE_QUIT'])
        
        self.Centre()
        
    def OnQuit(self, e):
        self.Close()

if __name__ == '__main__':
    app = wx.App()
    
    frame = MainFrame()
    frame.Show()

    app.MainLoop()
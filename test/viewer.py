class CTViewer:
    def __init__(self, vol):
        self.volume = vol
        self.index = 0
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.volume[self.index])
        self.fig.canvas.mpl_connect('key_press_event', self.update)
        
    def set_clim(self, lims):
        self.ax.images[0].set_clim(lims)
        self.fig.canvas.draw()
        
    def update(self, event):
        if event.key == 'right':
            self.next_slice()
        elif event.key == 'left':
            self.prev_slice()
        elif event.key == 'up':
            self.jump_fwd()
        elif event.key == 'down':
            self.jump_back()
        self.fig.canvas.draw()
        
    def next_slice(self):
        self.index = (self.index + 1) % self.volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(self.volume[self.index])
        
    def prev_slice(self):
        self.index = (self.index - 1) % self.volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(self.volume[self.index])
            
    def jump_fwd(self):
        self.index = (self.index + 5) % self.volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(self.volume[self.index])
             
    def jump_back(self):
        self.index = (self.index - 5) % self.volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(self.volume[self.index])

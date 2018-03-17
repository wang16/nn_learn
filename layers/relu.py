from layers.leaky_relu import LeakyReLU

class ReLU(LeakyReLU):
  def __init__(self, name, prev):
    LeakyReLU.__init__(self, name, 0, prev)




from gui.main.operation import Operation
from db import db

version = "0.0.4"

db.build()
app = Operation()
app.mainloop()

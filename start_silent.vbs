Set ws = CreateObject("Wscript.Shell")
ws.CurrentDirectory = "d:\杠杆择时策略"
ws.Run "cmd /c start_services.bat", 0, False

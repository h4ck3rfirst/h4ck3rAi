## Table of content

- virtual and private network btw kali and window
- run kali_server
- llm setting / gpt setting will run mcp_server

#### First need to create a virtual network bwteen virtual kali machine and window host machine 

go to setting of virtual box **select kali machine** -- **setting** -- **go to network setting** -- **enable two network adapte**r 1. for bridged network 2. host-only adapter
 
> save it

if restart you kali machine.  do run  `ifconfig` u will see the 2 adapter **eth1 and eth0** notes their is ip `192.168.56.*` 

**==if you see that success fucker success==**

### Running kali_server.py

run kali_server.py in side of virtual box with sudo perm
```
sudo python3 pathof-kali_server.py --ip 192.168.56.101 --port 5000 --debug
```
**another success fucker**

### llm and gpt setting on window

if you have llm running on your machine window machine

#### Connect with cluade/code/cursor/any gpt which connect with mcpserver

make enable a shared folder on kali and window witch share the same file at same time.

save that mcp-server.py at their 

downlaod python, claude/code on desktop then finish your login and other staff


**claude**
open/run claude and go to `setting` -- `developer` -- `click edit config` -- `open claude-desktop-config.json` file

```
{
  "mcpServers": {
    "kali-mcp": {
      "command": "python",
      "args": [
        "C:\\Users\\..\\..\\..path of shared folder which have ..\\..\\\mcp_server.py",
        "--server",
        "http://192.168.56.101:5000"
      ]
    }
  },
  "preferences": {
    "coworkScheduledTasksEnabled": false,
    "sidebarMode": "chat",
    "menuBarEnabled": false
  }
}
```

 save it 

 rerun the claude it will work if not readme again . then not working connect me  
 
 >nikhiljee23@proton.me





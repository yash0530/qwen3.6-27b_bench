import http.server, json, urllib.request
UP='http://127.0.0.1:8091/v1/messages'
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body=self.rfile.read(int(self.headers['Content-Length']))
        open('/tmp/cc_req_dump.json','wb').write(body)
        r=urllib.request.Request(UP,data=body,headers={'Content-Type':'application/json'})
        try:
            resp=urllib.request.urlopen(r).read()
            self.send_response(200)
        except Exception as e:
            resp=str(e).encode(); self.send_response(502)
        self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(resp)
    def log_message(self,*a): pass
http.server.HTTPServer(('127.0.0.1',8092),H).serve_forever()

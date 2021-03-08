# coding=UTF-8
# Inspiré du code webserver.py de Jon Berg , turtlemeat.com
# -*- py-which-shell: "python3"; -*-es

from pymongo import MongoClient
from bson.objectid import ObjectId
import json
import citeceer
import string,cgi,time
from os import curdir, sep
from http.server import BaseHTTPRequestHandler, HTTPServer

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        try:
            if self.path.endswith(".html"):
                f = open(curdir + sep + self.path)
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(bytes(f.read(), "utf-8"))
                f.close()
                return
            elif self.path.strip('/').isdigit():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                client = MongoClient()
                db=client['test']
                oai=db.oai     # possibly dbaz instead of oai
                Record = oai.find_one({'_id':ObjectId(citeceer.genKey(self.path.strip('/')))})
                self.wfile.write(bytes('<h1>' + self.path.lstrip('/0')+'</h1>', "utf-8"))
                self.wfile.write(bytes('<b>Title:</b><p>' + Record['dc:title'] + '</p>', "utf-8"))
                self.wfile.write(bytes('<b>Description:</b><p>' + Record['dc:description'] + '</p>', "utf-8"))
                self.wfile.write(bytes('<p><b>Authors:</b></p><p>' + ''.join(citeceer.getAuthors(Record) ) + '</p>', "utf-8"))
                self.wfile.write(bytes('<p><b>Date:</b></p><p>' + citeceer.getDate(Record) + '</p>', "utf-8"))
                self.wfile.write(bytes('<p><b>References:</b></p>' + citeceer.getRecordRef(Record, 'Ref', oai), "utf-8"))
                self.wfile.write(bytes('<p><b>Referers:</b></p>' + citeceer.getRecordRef(Record, 'RefBy', oai), "utf-8"))
            return
                
        except IOError as detail:
            self.send_error(404,'Fichier non trouvé: %s' % self.path)
            print("Erreur:", detail)
     

    def do_POST(self):
        global rootnode
        try:
            ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
            if ctype == 'multipart/form-data':
                query=cgi.parse_multipart(self.rfile, pdict)
            self.send_response(301)
            
            self.end_headers()
            upfilecontent = query.get('upfile')
            print("filecontent", upfilecontent[0])
            self.wfile.write(bytes("<HTML>POST OK.<BR><BR>", "utf-8"));
            self.wfile.write(bytes(upfilecontent[0], "utf-8"));
            
        except :
            pass

def main():
    try:
        server = HTTPServer(('', 8080), MyHandler)
        print('serveurweb démarré...')
        server.serve_forever()
    except KeyboardInterrupt:
        print('Interruption ^C, arrêt du serveur')
        server.socket.close()

if __name__ == '__main__':
    main()


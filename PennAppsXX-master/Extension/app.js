var http = require('http');
var fs = require("fs");
var url = require('url');

var server = http.createServer(function(req, res){
    fs.readFile('Result.txt', 'utf-8', (err, data) => {
        if (err) {
            res.writeHead(500);
            res.end(JSON.stringify(err));
        } else {
            res.writeHead(200, {'Content-Type': 'text/plain'})
            res.end(data);
        }
    });
});

server.listen(3000, '127.0.0.1');

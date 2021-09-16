#!/bin/bash
sudo printf '[mongodb-org-5.0]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/$releasever/mongodb-org/5.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-5.0.asc' > /etc/yum.repos.d/mongodb-org-5.0.repo
sudo yum install -y mongodb-org

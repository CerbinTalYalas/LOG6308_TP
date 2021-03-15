# Warning: keys are encoded in HEX but references in BD are in decimal; 

import re
import sys
from bson.objectid import ObjectId

def getRecordRef(Record, Direction, collection):
    relations = getRelations(Record)
    if(Record == []):
        return('')
    if(type(relations) == list):
        references = [ i['uri'] for i in relations if i['type'] == Direction ]
        referencesText = ''.join([ getCitation(i, collection) for i in references ])
        return(referencesText)

def getCitation(id, collection):
    cit = collection.find_one({'_id':ObjectId(genKey(str(id)))})
    if(cit == None):
        return(' ...[Missing ref], ')
    return('<li>' + ''.join(getAuthors(cit)) + ', (' + getDate(cit) + '). ' + '<a href="' + str(id).lstrip('0') + '">' + str(id).lstrip('0') + ":" +  cit['dc:title'] + '</a>')

def getAuthors(Record):
    if(not 'author' in Record):
        return('')
    if(type(Record['author']) is dict):
        return(Record['author']['name'])
    if(type(Record['author']) is list):
        return([x['name'] + ", " for x in Record['author']])
    return('')

def getDate(Record):
    if(Record['dc:date']):
        return(Record['dc:date'])
    return('')

def getRelations(Record):
    if 'relation' in Record.keys():
        return(Record['relation'])
    return([])

def genKey(kdec):
    khex = hex(int(kdec))[2:]   #  because url header are expected in decimal and keys are in HEX
    return(''.join([ '0' for i in range(0, 24 - len(khex)) ]) + khex)

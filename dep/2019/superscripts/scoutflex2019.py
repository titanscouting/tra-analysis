# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:21:31 2019

@author: creek
"""
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pprint
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape
import requests

def generate_team_report(team):
    doc = Document('basic')
    matches = team.reference.collection(u'matches').get()
    matchnums = []
    for match in matches:
        matchnums.append(match.id)
    with doc.create(Section('Qualification matches scouted')):
        for matchnum in matchnums:
            doc.append(str(matchnum))
    with doc.create(Section('Details')):
        hab = "Hab 1"
        balls = 42
        hatches = 0
        count = 0
        for match in matches:
            for analysis in match:
                if analysis.key().startswith('Quant'):
                    balls = balls + analysis['cargoBalls']
                    hatches = hatches + analysis['hatchPanels']
                    count = count + 1
                if analysis.key().startswith('Qual'):
                    strategy = analysis['StrategyType']
                    strongObject = analysis['TeleopStrongObject']
        if count > 0:
            doc.append("Average balls: " + str(float(balls)/count))
            doc.append("Average hatches: " + str(float(hatches)/count))
            doc.append("Strategy Type: " + str(strategy))
            doc.append("Strongest object in teleop: " + str(strongObject))


    doc.preamble.append(Command('title', team.id))
    doc.preamble.append(Command('author', 'Generated by Team 2022'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    doc.generate_pdf(filepath= str(team.id), clean_tex=False)

credential = credentials.Certificate('keys/firebasekey.json')

firebase_admin.initialize_app(credential)

db = firestore.Client()
teams_ref = db.collection(u'data').document(u'team-2022').collection(u'Central 2019')
teams = teams_ref.get()

for team in teams:
    generate_team_report(team)

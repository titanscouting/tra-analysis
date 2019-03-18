/* Set the width of the side navigation to 250px and the left margin of the page content to 250px and add a black background color to body */
function openNav() {
  document.getElementById("mySidenav").style.width = "250px";
  document.getElementById("main").style.marginLeft = "250px";
  document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
}

/* Set the width of the side navigation to 0 and the left margin of the page content to 0, and the background color of body to white */
function closeNav() {
  document.getElementById("mySidenav").style.width = "0";
  document.getElementById("main").style.marginLeft = "0";
  document.body.style.backgroundColor = "white";
}

window.onload = function() {
  document.getElementById('sideload').style.display = 'block';
  var config = {
    apiKey: "(insert the TitanScout Api Key Here)",
    authDomain: "titanscoutandroid.firebaseapp.com",
    databaseURL: "https://titanscoutandroid.firebaseio.com",
    projectId: "titanscoutandroid",
    storageBucket: "titanscoutandroid.appspot.com",
    messagingSenderId: "1097635313476"
  };
  //eventually find a less-jank way to do this tho
  firebase.initializeApp(config);
  firebase.auth().onAuthStateChanged(function(user) {
    if (user != null) {
      if (user.displayName != null) {
        document.getElementById('status').innerHTML = "You are signed in as: " + user.displayName;
      } else if (user.email != null) {
        document.getElementById('status').innerHTML = "You are signed in as: " + user.email;
      } else if (user.phoneNumber != null) {
        document.getElementById('status').innerHTML = "You are signed in as: " + user.phoneNumber;
      } else {
        document.getElementById('status').innerHTML = "You are signed in.";
      }
    } else {
      window.location.replace('../../');
    }
    teamAssoc = firebase.firestore().collection('UserAssociations').doc(user.uid);
    teamAssoc.get().then(function(doc) {
      if (doc.exists) {
        list = doc.data()
        teamNums = Object.keys(list)
        document.getElementById('tns').innerHTML = ""
        for (var i = 0; i < teamNums.length; i++) {
          document.getElementById('tns').innerHTML += "<option value='" + teamNums[i] + "'>" + teamNums[i] + "</option>"
        }
      } else {}
    }).then(function() {
      changeTeam(document.getElementById('tns').value)
    })
  });
}

function changeTeam(teamNum) {
  //remove all event listeners for element by cloning it
  var old_element = document.getElementById("mSelect");
  var new_element = old_element.cloneNode(true);
  old_element.parentNode.replaceChild(new_element, old_element);
  //ok back to normal stuff
  var user = firebase.auth().currentUser;
  ti = firebase.firestore().collection('teamData').doc("team-" + teamNum);
  currentComp = null;
  ti.get().then(function(doc) {
    if (doc.exists) {
      info = doc.data();
      currentComp = info['currentCompetition'];
    } else {
      alert("Something's wrong with firebase.");
      throw ("Something's wrong with firebase.");
    }
  }).then(function() {
    if (currentComp != null) {
      comps = firebase.firestore().collection('matchSignupsIndividual').doc(user.uid).collection("team-" + teamNum).doc(currentComp);
      comps.get().then(function(doc) {
        if (doc.exists) {
          var matches = doc.data();
          document.getElementById('mSelect').innerHTML = "";
          for (var i = 0; i < Object.keys(matches).length; i++) {
            name = Object.keys(matches)[i];
            series = matches[Object.keys(matches)[i]]['series'];
            completed = matches[Object.keys(matches)[i]]['completed'];
            if (!completed) {
              document.getElementById('mSelect').innerHTML += "<option value=" + name + series + ">" + name + "</option>"
            }
          }
          document.getElementById('mSelect').addEventListener("change", function() {
            updateForm(document.getElementById('mSelect').value, teamNum, currentComp);
          });
        }
      });
    }
  });
}

function lastWord(words) {
  var n = words.split(" ");
  return n[n.length - 1];
}

function firstWord(words) {
  var n = words.split(" ");
  return n[0];
}

function updateForm(locString, teamNum, competition) {
  seriesList = [];
  document.getElementById('FormData').innerHTML = ""
  loc = firebase.firestore().collection('appBuliding').document("team-" + teamNum).collection('competitions').document(competition).collection(lastWord(locString));
  loc.get().then(function(docs) {
    docs.forEach(function(doc) {
      seriesList.push(doc.data());
    });
    seriesList.sort(function(a, b) {
      return a.order - b.order;
    })
    for (var i = 0; i < seriesList.length; i++) {
      document.getElementById('FormData').innerHTML += "<h3>"
      seriesList[i].id + "</h3>";
      labels = Object.keys(seriesList[i].data());
      var index = labels.indexOf('order');
      if (index > -1) {
        labels.splice(index, 1);
      }
      var questions = [];
      for (var j = 0; j < labels.length; j++) {
        questions.push([labels[j], seriesList[i].data()[labels[j]]])
      }
      questions.sort(function(a, b) {
        return a[1].order - b[1].order;
      })
      for (var j = 0; j < questions.length; j++) {
        document.getElementById('FormData').innerHTML += "<div>";
        document.getElementById('FormData').innerHTML += questions[j][1]['title'];
        if (questions[j][1]['type'] = 'shortText') {
          document.getElementById('FormData').innerHTML += "<input id=''" + questions[j][0] + "' type='text'></input>";
        } else if (questions[j][1]['type'] = 'longText') {
          document.getElementById('FormData').innerHTML += "<textarea id=''" + questions[j][0] + "' rows='4' cols='50''></textarea>";
        } else if (questions[j][1]['type'] = 'numerical') {
          document.getElementById('FormData').innerHTML += "<span id='" + questions[j][0] + "'><input type='button' onclick='dec(" + questions[j][0] + ")' value='-'></input>" + (questions[j][1]['default']).toString() + "<input type='button' onclick='inc(" + questions[j][0] + ")' value='+'></input></span>";
        } else if (questions[j][1]['type'] = 'range') {
          document.getElementById('FormData').innerHTML += "&nbsp;&nbsp;" + questions[j][1]['min']['text'] + "&nbsp;&nbsp;";
          document.getElementById('FormData').innerHTML += "<input type='range' min='" + questions[j][1]['min']['val'] + "' max='" + questions[j][1]['max']['val'] + "'>";
          document.getElementById('FormData').innerHTML += "&nbsp;&nbsp;" + questions[j][1]['max']['text'];
        } else if (questions[j][1]['type'] = 'segment') {
          document.getElementById('FormData').innerHTML += "<div id='" + questions[j][0] + "'>"
          for (var k = 0; k < questions[j][1]['elements'].length; k++) {
            //// TODO: replace with real buttons for good styling
            document.getElementById('FormData').innerHTML += questions[j][1]['elements'][k];
            document.getElementById('FormData').innerHTML += "<input type='radio' name='" + questions[j][0] + "' value=" + questions[j][1]['elements'][k] + "></input>"
          }
          document.getElementById('FormData').innerHTML += "</div>"
        }
        document.getElementById('FormData').innerHTML += "</div>";
      }
    }
    document.getElementById('FormData').innerHTML += "<input type='button' onclick=subReport(" + teamNum + "," + competition + "," + firstWord(locString) + ") value='Submit'>"
  });
}

function dec(id) {
  document.getElementById(id).innerHTML = (parseInt(document.getElementById(id).innerHTML) - 1)).toString()
}

function inc(id) {
  document.getElementById(id).innerHTML = (parseInt(document.getElementById(id).innerHTML) + 1)).toString()
}

function subReport(team, comp, matchNum) {
  var push = {}
  var x = document.getElementById('FormData').children;
  for (var i = 0; i < x.length; i++) {
    if (x[i].children[0].tagName == "INPUT") {
      push[x[i].children[0].id] = x[i].children[0].tagName;
    } else if (x[i].children[0].tagName == "TEXTAREA") {
      push[x[i].children[0].id] = x[i].children[0].innerHTML;
    } else if (x[i].children[0].tagName == "SPAN") {
      push[x[i].children[0].id] = x[i].children[0].innerText;
    }else if (x[i].children[0].tagName == "DIV") {
      var name=x[i].children[0].id;
      push[name]=document.querySelector('input[name="'+name+'"]:checked').value;
    }
  }
  var user = firebase.auth().currentUser;
  firebase.firestore().collection("teamData").doc('team-' + team).collection('scouts').doc(user.uid).collection(comp).doc("team-" + scoutedTeamNumber + matchNum).set(push, {
    merge: true
  }).then(function () {
    firebase.firestore().collection("data").doc('team-' + team).collection(comp).doc("team-" + scoutedTeamNumber).collection('matches').doc('match-' + matchNum).set(push, {
      merge: true
    });
  });
}

/*

function subRes() {
  firebase.firestore().collection('teamData').doc('team-' + document.getElementById('tns').value).get().then(function(doc) {
    if (doc.exists) {
      comp = doc.data()['currentCompetition'];
    }
  }).then(function() {


    var user = firebase.auth().currentUser;
    push = {}
    push['match'] = 'match-' + document.getElementById('mn').value
    push['teamDBRef'] = 'team-' + document.getElementById('tsn').value
    push['speed'] = document.getElementById('speed').value
    push['sandstormCross'] = document.getElementById('SCross').value
    push['strategy'] = document.getElementById('strat').value
    push['contrubution'] = document.getElementById('contrib').value
    push['startingHatch'] = document.getElementById('habs').value
    push['size'] = document.getElementById('egs').value

    push['fillChoice'] = document.getElementById('SFill').value
    push['functional'] = document.getElementById('DOA').value
    push['strongMedium'] = document.getElementById('SSO').value
    push['sandstormCrossBonus'] = document.getElementById('SCross').value

    push['fillChoiceTeleop'] = document.getElementById('TFill').value
    push['strongMediumTeleop'] = document.getElementById('TSO').value

    push['cargoSuccessTeleop'] = document.getElementById('CSSR').value
    push['hiRocketSuccessTeleop'] = document.getElementById('HRSR').value
    push['lowRocketSuccessTeleop'] = document.getElementById('LRSR').value

    push['endingHab'] = document.getElementById('HAB').value

    firebase.firestore().collection("teamData").doc('team-' + document.getElementById('tns').value).collection('scouts').doc(user.uid).collection(comp).doc("team-" + document.getElementById('tsn').value + "-match-" + document.getElementById('mn').value).set(push, {
      merge: true
    })
  }).then(function() {
      var user = firebase.auth().currentUser;
      push = {}
      push['match'] = 'match-' + document.getElementById('mn').value
      push['teamDBRef'] = 'team-' + document.getElementById('tsn').value
      push['speed'] = document.getElementById('speed').value
      push['sandstormCross'] = document.getElementById('SCross').value
      push['strategy'] = document.getElementById('strat').value
      push['contrubution'] = document.getElementById('contrib').value
      push['startingHatch'] = document.getElementById('habs').value
      push['size'] = document.getElementById('egs').value

      push['fillChoice'] = document.getElementById('SFill').value
      push['functional'] = document.getElementById('DOA').value
      push['strongMedium'] = document.getElementById('SSO').value
      push['sandstormCrossBonus'] = document.getElementById('SCross').value

      push['fillChoiceTeleop'] = document.getElementById('TFill').value
      push['strongMediumTeleop'] = document.getElementById('TSO').value

      push['cargoSuccessTeleop'] = document.getElementById('CSSR').value
      push['hiRocketSuccessTeleop'] = document.getElementById('HRSR').value
      push['lowRocketSuccessTeleop'] = document.getElementById('LRSR').value

      push['endingHab'] = document.getElementById('HAB').value
      firebase.firestore().collection("data").doc('team-' + document.getElementById('tns').value).collection(comp).doc("team-" + document.getElementById('tsn').value).collection('matches').doc('match-' + document.getElementById('mn').value).set(push, {
        merge: true
      })
    }

  ).then(function() {
    alert('Submitted!')
    setTimeout(function() {
      window.location.href = '../scout';
    }, 500);

  })
}*/

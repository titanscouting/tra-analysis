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
  document.getElementById("matchTable").innerHTML = `<tr>
    <td class="neu">Match Number</td>
    <td class="neu">Series</td>
    <td class="blue">Far Blue</td>
    <td class="blue">Mid Blue</td>
    <td class="blue">Near Blue</td>
    <td class="red">Far Red</td>
    <td class="red">Mid Red</td>
    <td class="red">Near Red</td>
  </tr>`;
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
    cci = firebase.firestore().collection('matchSignupsTeam').doc("team-" + teamNum).collection('competitions').doc(currentComp);
    cci.get().then(function(doc) {
      if (doc.exists) {
        compInfo = doc.data();
        matches = Object.keys(compInfo);
        matches.sort();
        var nr = [],
          mr = [],
          fr = [],
          nb = [],
          mb = [],
          fb = [];
        for (var i = 0; i < matches.length; i++) {
          mi = compInfo["match-" + (i + 1).toString()]
          //sets up the table lists. i really hope it doesn't break.
          for (var j = 0; j < 2; j++) {
            if (mi['far-blue']['series-' + (j + 1).toString()] != null) {
              fb.push(mi['far-blue']['series-' + (j + 1).toString()]);
            } else {
              fb.push("<span onclick='addMatch(" + (i + 1).toString() + "," + (j + 1).toString() + ",'far-blue')'>open</span>");
            }
            if (mi['mid-blue']['series-' + (j + 1).toString()] != null) {
              mb.push(mi['mid-blue']['series-' + (j + 1).toString()]);
            } else {
              mb.push("<span onclick='addMatch(" + (i + 1).toString() + "+", (j + 1).toString() + ",'mid-blue')'>open</span>");
            }
            if (mi['near-blue']['series-' + (j + 1).toString()] != null) {
              nb.push(mi['near-blue']['series-' + (j + 1).toString()]);
            } else {
              nb.push("<span onclick='addMatch(" + (i + 1).toString() + "," + (j + 1).toString() + ",'near-blue')'>open</span>");
            }
            if (mi['far-red']['series-' + (j + 1).toString()] != null) {
              fr.push(mi['far-red']['series-' + (j + 1).toString()]);
            } else {
              fr.push("<span onclick='addMatch(" + (i + 1).toString() + "," + (j + 1).toString() + ",'far-red')'>open</span>");
            }
            if (mi['mid-red']['series-' + (j + 1).toString()] != null) {
              mr.push(mi['mid-red']['series-' + (j + 1).toString()]);
            } else {
              mr.push("<span onclick='addMatch(" + (i + 1).toString() + "," + (j + 1).toString() + ",'mid-red')'>open</span>");
            }
            if (mi['near-red']['series-' + (j + 1).toString()] != null) {
              nr.push(mi['near-red']['series-' + (j + 1).toString()]);
            } else {
              nr.push("<span onclick='addMatch(" + (i + 1).toString() + "," + (j + 1).toString() + ",'near-red')'>open</span>")
            }
          }
          var outstr = "";
          outstr += "<tr><td rowspan='2' class='neu'>Quals " + (i + 1).toString() + "</td>";
          outstr += "<td class='neu'>Series 1</td>";
          outstr += "<td class='blue'>" + fb[0] + "</td>";
          outstr += "<td class='blue'>" + mb[0] + "</td>";
          outstr += "<td class='blue'>" + nb[0] + "</td>";
          outstr += "<td class='red'>" + fr[0] + "</td>";
          outstr += "<td class='red'>" + mr[0] + "</td>";
          outstr += "<td class='red'>" + nr[0] + "</td>";
          outstr += "</tr>"
          for (var k = 1; k < 2; k++) {
            outstr += "<tr>";
            outstr += "<td class='neu'>Series " + (k + 1).toString() + "</td>";
            outstr += "<td class='blue'>" + fb[k] + "</td>";
            outstr += "<td class='blue'>" + mb[k] + "</td>";
            outstr += "<td class='blue'>" + nb[k] + "</td>";
            outstr += "<td class='red'>" + fr[k] + "</td>";
            outstr += "<td class='red'>" + mr[k] + "</td>";
            outstr += "<td class='red'>" + nr[k] + "</td>";
            outstr += "</tr>"
          }
          document.getElementById('matchTable').innerHTML += outstr;
        }
      }
    });
  });
}

function addMatch(matchNum, seriesNum, position) {
  var success = false;
  var teamNum = document.getElementById('tns').value
  var user = firebase.auth().currentUser;
  var name = "anon"
  if (user.displayName != null) {
    name = user.displayName;
  } else if (user.email != null) {
    name = user.email;
  } else if (user.phoneNumber != null) {
    name = user.phoneNumber;
  }
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
    cci = firebase.firestore().collection('matchSignupsTeam').doc("team-" + teamNum).collection('competitions').doc(currentComp);
    cci.get().then(function(doc) {
      if (doc.exists) {
        info = doc.data();
        match = info["match-" + matchNum.toString()];
        pos = match[position];
        occ = pos["series-" + seriesNum.toString()];
        if (occ == null) {
          info["match-" + matchNum.toString()][position]["series-" + seriesNum.toString()] = name;
          firebase.firestore().collection('matchSignupsTeam').doc("team-" + teamNum).collection('competitions').doc(currentComp).set(info)
          success = true;
        } else {
          alert(occ + "has added that match first.")
          setTimeout(function() {
            window.location.href = '../signUps';
          }, 500);
        }
      }
    }).then(function() {
      if (success) {
        ti = firebase.firestore().collection('matchSignupsIndividual').doc(user.uid).collection("team-" + teamNum).doc(currentComp);
        label="match-" + matchNum.toString()+" "+position
        push = {label: {
            'completed': false,
            'series': seriesNum.toString()
          }
        }
        cityRef.set(push, {
          merge: true
        }), then(function() {
          alert('Added!')
          setTimeout(function() {
            window.location.href = '../signUps';
          }, 500);
        });
      }
    });
  });
}

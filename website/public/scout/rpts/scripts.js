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
    })
  });
}

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
    push['strategy']= document.getElementById('strat').value
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
      push['strategy']= document.getElementById('strat').value
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
    setTimeout(function(){ window.location.href = '../scout'; }, 500);

  })
}

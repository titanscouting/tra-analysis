/* Set the width of the side navigation to 250px and the left margin of the page content to 250px and add a black background color to body */
function openNav() {
  document.getElementById("mySidenav").style.width = "250px";
  document.getElementById("main").style.marginLeft = "250px";
  document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
  for (var i = 0; i < document.getElementsByClassName("btn").length; i++) {
    document.getElementsByClassName("btn")[i].style.backgroundColor = "rgba(0,0,0,.2)"
  }
}

/* Set the width of the side navigation to 0 and the left margin of the page content to 0, and the background color of body to white */
function closeNav() {
  document.getElementById("mySidenav").style.width = "0";
  document.getElementById("main").style.marginLeft = "0";
  document.body.style.backgroundColor = "white";
  for (var i = 0; i < document.getElementsByClassName("btn").length; i++) {
    document.getElementsByClassName("btn")[i].style.backgroundColor = "buttonface"
  }
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
  firebase.initializeApp(config);
  firebase.auth().onAuthStateChanged(function(user) {
    if (user != null) {
      if (user.displayName != null) {
        document.getElementById('status').innerHTML = "You are signed in as: " + user.displayName;
        document.getElementById('newDN').innerHTML = user.displayName;
      } else if (user.email != null) {
        document.getElementById('status').innerHTML = "You are signed in as: " + user.email;
      } else if (user.phoneNumber != null) {
        document.getElementById('status').innerHTML = "You are signed in as: " + user.phoneNumber;
      } else {
        document.getElementById('status').innerHTML = "You are signed in.";
      }
      if (user.email != null) {
        document.getElementById('newEM').innerHTML = user.email;
      }
    } else {
      window.location.replace('../');
    }
    firebase.firestore.settings({timestampsInSnapshots: true})
    teamAssoc = firebase.firestore().collection('UserAssociations').doc(user.uid);
    teamAssoc.get().then(function(doc) {
      if (doc.exists) {
        list = doc.data()
        teamNums = Object.keys(list)
        document.getElementById('teammem').innerHTML = ""
        for (var i = 0; i < teamNums.length; i++) {
          document.getElementById('teammem').innerHTML += "<tr><td>" + teamNums[i] + "</td><td>" + list[teamNums[i]] + "</td></tr>"
        }
      } else {
        document.getElementById('teammem').innerHTML = "<tr><td>You are not part of any teams</td></tr>"
      }
    })
  });
}

function cnt(tn) {
  user=firebase.auth().currentUser;
  push={}
  push[tn]='captian'
  firebase.firestore().collection("UserAssociations").doc(user.uid).set(push, {
    merge: true
  }).then(function() {
    teamAssoc = firebase.firestore().collection('UserAssociations').doc(user.uid)
    teamAssoc.get().then(function(doc) {
      if (doc.exists) {
        list = doc.data()
        teamNums = Object.keys(list)
        document.getElementById('teammem').innerHTML = ""
        for (var i = 0; i < teamNums.length; i++) {
          document.getElementById('teammem').innerHTML += "<tr><td>" + teamNums[i] + "</td><td>" + list[teamNums[i]] + "</td></tr>"
        }
      } else {
        document.getElementById('teammem').innerHTML = "<tr><td>You are not part of any teams</td></tr>"
      }
    })
  })
}
function checkKeyMatch(dt,tn,key){
    for(i=0; i<Object.keys(dt).length; i++){
        if (Object.keys(dt)[i]=="code-"+key){
            if (dt[Object.keys(dt)[i]]==tn){
                return true
            }
        }
    }
    return false
}
function reqjt(tn,tc){
  user=firebase.auth().currentUser;
  dict={}
  firebase.firestore().collection('teamData').doc('joinCodes').get().then(function(doc){
    if (doc.exists) {
       dict=doc.data();
   } else {
       // doc.data() will be undefined in this case
       console.log("No such document!");
   }
  });
  if (checkKeyMatch(dict,tn,tc)){
    push={};
    push[tn]='scout';
    firebase.firestore().collection("UserAssociations").doc(user.uid).set(push, {
      merge: true
    }).then(function(doc) {
      if (doc.exists) {
        list = doc.data()
        teamNums = Object.keys(list)
        document.getElementById('teammem').innerHTML = ""
        for (var i = 0; i < teamNums.length; i++) {
          document.getElementById('teammem').innerHTML += "<tr><td>" + teamNums[i] + "</td><td>" + list[teamNums[i]] + "</td></tr>"
        }
      } else {
        document.getElementById('teammem').innerHTML = "<tr><td>You are not part of any teams</td></tr>"
      }
    })
  }else{
    alert("You don't have a correct join key. Please check it and try again.")
  }
}

function signout() {
  var user = firebase.auth().currentUser;
  firebase.auth().signOut().then(
    window.location.href = '../');
}

function deleteAccount() {
  try {
    firebase.auth().currentUser.delete().then(
      window.location.href = '../');
  } catch (error) {
    if (error.code == 'auth/requires-recent-login') {
      alert("Please sign in again to delete your account.")
      window.location.href = '../';
    }
  }
}

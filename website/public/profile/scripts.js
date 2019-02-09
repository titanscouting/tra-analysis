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
  });
}

function signout() {
  var user = firebase.auth().currentUser;
  firebase.auth().signOut()
  window.location.href = '../';
}

function deleteAccount() {
  try {
    firebase.auth().currentUser.delete()
    window.location.href = '../';
  } catch (error) {
    if (error.code == 'auth/requires-recent-login') {
      alert("Please sign in again to delete your account.")
      window.location.href = '../';
    }
  }
}

function updun() {
  var user = firebase.auth().currentUser;
  user.updateProfile({
    displayName: document.getElementById('newDN').innerHTML,
  }).then(function() {
    document.getElementById('newDN').innerHTML = firebase.auth().currentUser.displayName;
    document.getElementById('status').innerHTML = "You are signed in as: " + firebase.auth().currentUser.displayName;
  }).catch(function(error) {
    alert("there was a problem: " + error)
  });
}

function updem() {
  var user = firebase.auth().currentUser;
  user.updateEmail(document.getElementById('newEM').innerHTML).then(function() {
    if (user.displayName != null) {
      document.getElementById('status').innerHTML = "You are signed in as: " + user.displayName;
      document.getElementById('newDN').innerHTML = user.displayName;
    } else if (user.email != null) {
      document.getElementById('status').innerHTML = "You are signed in as: " + user.email;
    }
  }).catch(function(error) {
    if (error.code == 'auth/requires-recent-login') {
      alert("Please sign in again to delete your account.")
      window.location.href = '../';
    } else {
      alert("there was a problem: " + error)
    }
  });
}

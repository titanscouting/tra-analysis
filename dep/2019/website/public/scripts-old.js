var handleSignedInUser = function(user) {
  document.getElementById("mainhead").innerHTML = "TitanScout- User Info";
  if (user.displayName != null) {
    document.getElementById('status').innerHTML = "You are signed in as: " + user.displayName;
  } else if (user.email != null) {
    document.getElementById('status').innerHTML = "You are signed in as: " + user.email;
  } else if (user.phoneNumber != null) {
    document.getElementById('status').innerHTML = "You are signed in as: " + user.phoneNumber;
  } else {
    document.getElementById('status').innerHTML = "You are signed in.";
  }
  document.getElementById('signout').style.display = 'inline-block';
  document.getElementById('updpi').style.display = 'inline-block';
  document.getElementById('deleteacc').style.display = 'inline-block';
  document.getElementById('profileupd').style.display = 'none';
  document.getElementById('sideload').style.display = 'block';
}
var handleSignedOutUser = function() {
  document.getElementById("mainhead").innerHTML = "TitanScout- Sign In";
  document.getElementById('status').innerHTML = "You are not signed in.";
  document.getElementById('signout').style.display = 'none';
  document.getElementById('updpi').style.display = 'none';
  document.getElementById('deleteacc').style.display = 'none';
  document.getElementById('profileupd').style.display = 'none';
  document.getElementById('sideload').style.display = 'none';
  ui.start('#firebaseui-auth-container', uiConfig);
};

// Initialize the FirebaseUI Widget using Firebase.
var ui = new firebaseui.auth.AuthUI(firebase.auth());

// The start method will wait until the DOM is loaded.
ui.start('#firebaseui-auth-container', uiConfig);
var deleteAccount = function() {
  try {
    firebase.auth().currentUser.delete()
    handleSignedOutUser()
  } catch (error) {
    if (error.code == 'auth/requires-recent-login') {
      // The user's credential is too old. She needs to sign in again.
      signout()
      // The timeout allows the message to be displayed after the UI has
      // changed to the signed out state.
      setTimeout(function() {
        alert('Please sign in again to delete your account.');
      }, 1);
    }
  }
};

function signout() {
  var user = firebase.auth().currentUser;
  firebase.auth().signOut()
  handleSignedOutUser()
}

function loadupdpi() {
  if (firebase.auth().currentUser != null) {
    document.getElementById('profileupd').style.display = 'block';
  } else {
    setTimeout(function() {
      alert('Please sign in to change your account info.');
    }, 1);
    handleSignedOutUser();
  }
}

function upProfileInfo() {
  if (firebase.auth().currentUser != null) {
    var user = firebase.auth().currentUser;
    var newDN = document.getElementById('newDN').value;
    var newEM = document.getElementById('newEM').value;
    var newPP = document.getElementById('newPP').value;
    var si = true
    if (newDN != '' && newDN != user.displayName) {
      if (newPP != '' && newPP != user.photoURL) {
        try {
          user.updateProfile({
            displayName: newDN,
            photoURL: newPP
          });
        } catch (error) {
          if (error.code == 'auth/requires-recent-login') {
            si = false;
            // The user's credential is too old. She needs to sign in again.
            signout()
            // The timeout allows the message to be displayed after the UI has
            // changed to the signed out state.
            setTimeout(function() {
              alert('Please sign in again to delete your account.');
            }, 1);
          } else {
            alert("An error occurred: " + error)
          }
        }
      } else {
        try {
          user.updateProfile({
            displayName: newDN
          });
        } catch (error) {
          if (error.code == 'auth/requires-recent-login') {
            si = false;
            // The user's credential is too old. She needs to sign in again.
            signout()
            // The timeout allows the message to be displayed after the UI has
            // changed to the signed out state.
            setTimeout(function() {
              alert('Please sign in again to delete your account.');
            }, 1);
          } else {
            alert("An error occurred: " + error)
          }
        }
      }
    } else {
      if (newPP != '' && newPP != user.photoURL) {
        try {
          user.updateProfile({
            photoURL: newPP
          });
        } catch (error) {
          if (error.code == 'auth/requires-recent-login') {
            si = false;
            // The user's credential is too old. She needs to sign in again.
            signout()
            // The timeout allows the message to be displayed after the UI has
            // changed to the signed out state.
            setTimeout(function() {
              alert('Please sign in again to delete your account.');
            }, 1);
          } else {
            alert("An error occurred: " + error)
          }
        }
      }
    }
    if (newEM != '' && newEM != user.email) {
      try {
        user.updateEmail(newEM)
      } catch (error) {
        si = false;
        if (error.code == 'auth/requires-recent-login') {
          // The user's credential is too old. She needs to sign in again.
          signout()
          // The timeout allows the message to be displayed after the UI has
          // changed to the signed out state.
          setTimeout(function() {
            alert('Please sign in again to delete your account.');
          }, 1);
        } else {
          alert("An error occurred: " + error)
        }
      }
    }
    if (si) {
        setTimeout(function(){handleSignedInUser(user);},1)
    }

  } else {
    setTimeout(function() {
      alert('Please sign in to change your account info.');
    }, 1);
    handleSignedOutUser();
  }
}
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
window.onload=function(){
  if(firebase.auth().currentUser!=null){
    handleSignedInUser(firebase.auth().currentUser)
  }
}

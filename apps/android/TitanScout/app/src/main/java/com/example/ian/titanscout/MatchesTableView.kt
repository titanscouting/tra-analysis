package com.example.ian.titanscout

import android.graphics.Color
import android.os.Bundle
import android.support.design.widget.FloatingActionButton
import android.support.v7.app.AlertDialog
import android.support.v7.app.AppCompatActivity;

import com.google.firebase.firestore.FirebaseFirestore
import org.json.JSONObject
import com.google.zxing.WriterException
import android.graphics.Bitmap
import android.widget.*
import net.glxn.qrgen.android.QRCode


class MatchesTableView : AppCompatActivity() {


    var shouldShow = true
    // Reference the database to be used in the rest of the class.
    val db = FirebaseFirestore.getInstance()
    var alias = ""
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_matches_table_view)
//        setSupportActionBar(toolbar)

        var matches = arrayOf<Match>()
        val TAG = "MainActivity"
        val docRef = db.collection("appBuilding").document("schedule")
        docRef.get().addOnSuccessListener { documentSnapshot ->
            val stringData = documentSnapshot.data.toString()

            // Get data from the database and process it into an array of Matches.
            val schedule = Response(stringData).getJSONArray("matches")
            for (i in 0..(schedule.length() - 1)) {
                val item = schedule.getJSONObject(i)
                val reds = Response(item["RED"].toString())
                val blues = Response(item["BLUE"].toString())
                val redTeams = getTeamArrayFromJSON(reds, "RED")
                val blueTeams = getTeamArrayFromJSON(blues, "BLUE")
                matches += Match(i+1, redTeams, blueTeams)
            }

            // update the user's alias
            alias = intent.getStringExtra("alias")
            updateAlias(alias)


            val listView = findViewById<ListView>(R.id.match_list_view)

            val listItems = arrayOfNulls<String>(matches.size)

            for (i in 0 until matches.size) {
                val match = matches[i]
                listItems[i] = "Match " + match.ind
            }


//            val adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, listItems)
//            listView.adapter = adapter

            val adapter = MatchAdapter(this, matches)
            listView.adapter = adapter

        }

        val fab = findViewById<FloatingActionButton>(R.id.fab)
        fab.setImageResource(R.drawable.qrcodeicon)
        fab.setColorFilter(Color.parseColor("#FFFFFF"))


        fab.setOnClickListener { view ->

            // QR Button pressed

            if (shouldShow) {
                try {

                    val bitmap = TextToImageEncode(intent.getStringExtra("auth"))


                    findViewById<ImageView>(R.id.imageView).setImageBitmap(bitmap)

                } catch (e: WriterException) {
                    e.printStackTrace()
                }

            } else {
                findViewById<ImageView>(R.id.imageView).setImageResource(android.R.color.transparent)
            }

            shouldShow = !shouldShow
        }

        findViewById<Button>(R.id.changeAliasButton).setOnClickListener {
            showCreateCategoryDialog()
        }


    }


    fun TextToImageEncode(text: String): Bitmap {

        // generate a QR code from the given text.
        return QRCode.from(text).withSize(1000, 1000).bitmap()


    }

    // From https://code.luasoftware.com/tutorials/android/android-text-input-dialog-with-inflated-view-kotlin/
    fun showCreateCategoryDialog() {
        val context = this
        val builder = AlertDialog.Builder(context)
        builder.setTitle("Change Alias")

        // https://stackoverflow.com/questions/10695103/creating-custom-alertdialog-what-is-the-root-view
        // Seems ok to inflate view with null rootView
        val view = layoutInflater.inflate(R.layout.dialog_new_category, null)

        val categoryEditText = view.findViewById(R.id.categoryEditText) as EditText

        builder.setView(view)

        // set up the ok button
        builder.setPositiveButton(android.R.string.ok) { dialog, p1 ->
            val newCategory = categoryEditText.text
            var isValid = true
            if (newCategory.isBlank()) {
                categoryEditText.error = "Some String"
                isValid = false
            }

            if (isValid) {
                // do something
                updateAlias(categoryEditText.text.toString())
            }

            if (isValid) {
                dialog.dismiss()
            }
        }

        builder.setNegativeButton(android.R.string.cancel) { dialog, p1 ->
            dialog.cancel()
        }

        builder.show();
    }

    fun updateAlias(withString: String) {
        alias = withString
        val str = "Hello! I'm " + withString
        findViewById<TextView>(R.id.aliasTextView).text = str
    }

    // When given a JSON containing teams and scouts like this : {"team-4096":[],"team-101":["scoutName"],"team-4292":["ScoutName", "ScoutName32"]}},
    // The function will convert it to a Team Array
    fun getTeamArrayFromJSON(json: JSONObject, forColor:String) : Array<Team> {
        var teams = arrayOf<Team>()
        val keys = json.keys()
        while (keys.hasNext()) {
            // get the key
            val key = keys.next()

            // Manually parse the string into an array of strings.
            var vs = json.get(key).toString()
            var scouts = arrayOf<String>()
            vs = vs.substring(1,vs.length-1)
            if (!vs.equals("") && !vs.contains(",")) {
                scouts += vs.substring(1,vs.length-1)
            }
            for (str in vs.split(",")) {
                if (str.length>2) {
                    scouts += str.substring(1,str.length-1)
                }
            }

            teams += Team(key.substring(5), forColor, scouts)

        }
        return teams

    }

}

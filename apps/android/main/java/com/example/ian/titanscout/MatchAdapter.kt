package com.example.ian.titanscout

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.ProgressBar
import android.widget.TextView

class MatchAdapter(private val context: Context,
                   private val dataSource: Array<Match>) : BaseAdapter() {

    private val inflater: LayoutInflater = context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater

    //1
    override fun getCount(): Int {
        return dataSource.size
    }

    //2
    override fun getItem(position: Int): Any {
        return dataSource[position]
    }

    //3
    override fun getItemId(position: Int): Long {
        return position.toLong()
    }

    //4
    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        // Get view for row item
        val rowView = inflater.inflate(R.layout.list_item, parent, false)


        // Get title element
        val titleTextView = rowView.findViewById(R.id.titleTV) as TextView

        // Get subtitle element
        val subtitleTextView = rowView.findViewById(R.id.subtitle) as TextView

        // Get progressBar element
        val progressBar = rowView.findViewById(R.id.progressBar) as ProgressBar

        val match = getItem(position) as Match

        val str = "Match " + match.ind
        titleTextView.text = str
        val str2 = match.getScouts().toString() + " of 6"
        subtitleTextView.text = str2
        progressBar.progress = (1.0 * progressBar.max * match.getScouts() / (6.0)).toInt()

        return rowView
    }


}
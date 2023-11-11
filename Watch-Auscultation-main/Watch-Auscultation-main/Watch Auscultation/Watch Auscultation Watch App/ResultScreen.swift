//
//  ResultScreen.swift
//  Watch Auscultation Watch App
//
//  Created by Rucha Nandha on 11/4/23.
//

import SwiftUI

struct ResultScreen: View {
    @Environment(\.presentationMode) var presentationMode

    
    var body: some View {
        NavigationStack {
            VStack {
                Text("You tested negative!")
                    .font(.headline)
                    .padding()
                
                NavigationLink("Dismiss", destination: ContentView(isRecording: .constant(true)))
                
            }
        }
    }
}

#Preview {
    ResultScreen()
}

//
//  RecordScreen.swift
//  Watch Auscultation Watch App
//
//  Created by Rucha Nandha on 11/4/23.
//

import SwiftUI
import AVFoundation

struct RecordScreen: View {
    @Binding var isRecording: Bool
    @State private var audioRecorder: AVAudioRecorder?
    @State private var navigate = false
    @State private var isLoading = false

    var body: some View {
        NavigationView {
            VStack {
                Spacer()
               Image(systemName: "heart.fill")
                    .foregroundStyle(.red)
                    .font(.largeTitle)
                    .keyframeAnimator(initialValue: AnimationValues()) { content, value in
                        content
                            .foregroundStyle(.white)
                            .rotationEffect(value.angle)
                            .scaleEffect(value.scale)
                            .scaleEffect(y: value.verticalStretch)
                            .offset(y: value.verticalTranslation)
                    } keyframes: { _ in
                        KeyframeTrack(\.scale) {
                            LinearKeyframe(1.0, duration: 0.36)
                            SpringKeyframe(1.5, duration: 0.8, spring: .bouncy)
                            SpringKeyframe(1.0, spring: .bouncy)
                        }
                    }
                Spacer()
                Text("Lay your hand on your heart...")
                    .font(.caption2)
                if isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(1.5)
                } else {
                    Button("Go to Detail Screen") {
                        isLoading = true
                        // Simulate a delay for loading
                        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                            isLoading = false
                            navigate = true
                        }
                    }
                    
                }
                
                NavigationLink(destination: ResultScreen(), isActive: $navigate) {
                    EmptyView()
                    
                    
                    
                    /*  if isRecording {
                     
                     //Button(action: {
                     //    stopRecording()
                     //}) {
                     
                     
                     // }
                     } else {
                     Text("")
                     .font(.caption2)
                     }*/
                    
                    
                }
                .frame(width: 0, height: 0)
                .hidden()
                .edgesIgnoringSafeArea(.vertical)
            }
        }
    }

    func startRecording() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default)
            try audioSession.setActive(true)
        } catch {
            print("Error setting up audio session: \(error)")
        }
        
        let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let audioFileURL = documentDirectory.appendingPathComponent("recording.wav")

        let audioSettings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 44100.0,
            AVNumberOfChannelsKey: 2,
            AVEncoderBitRateKey: 25600,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: audioFileURL, settings: audioSettings)
            audioRecorder?.record()
            isRecording = true
        } catch {
            print("Error setting up audio recorder: \(error)")
        }
    }

    func stopRecording() {
        audioRecorder?.stop()
        isRecording = false
    }
}


#Preview {
    RecordScreen(isRecording: .constant(true))
}

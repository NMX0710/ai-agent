import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: SyncViewModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Backend") {
                    TextField("Base URL", text: $viewModel.configuration.baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)

                    TextField("Bridge Token", text: $viewModel.configuration.bridgeToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    TextField("User ID", text: $viewModel.configuration.userID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Apple Health") {
                    HStack {
                        Text("Authorization")
                        Spacer()
                        Text(viewModel.authorizationStatusText)
                            .foregroundStyle(.secondary)
                    }

                    Button("Request Health Access") {
                        Task {
                            await viewModel.requestAuthorization()
                        }
                    }
                }

                Section("Sync") {
                    Button {
                        Task {
                            await viewModel.syncNow()
                        }
                    } label: {
                        if viewModel.isSyncing {
                            HStack {
                                ProgressView()
                                Text("Syncing...")
                            }
                        } else {
                            Text("Sync Now")
                        }
                    }
                    .disabled(viewModel.isSyncing)

                    if let summary = viewModel.lastSummary {
                        LabeledContent("Last Result", value: summary.title)
                        LabeledContent("Processed", value: "\(summary.processed)")
                        LabeledContent("Succeeded", value: "\(summary.succeeded)")
                        LabeledContent("Failed", value: "\(summary.failed)")
                        if !summary.message.isEmpty {
                            Text(summary.message)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Text(summary.timestamp.formatted(date: .abbreviated, time: .standard))
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("No sync has run yet.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Health Sync")
        }
    }
}

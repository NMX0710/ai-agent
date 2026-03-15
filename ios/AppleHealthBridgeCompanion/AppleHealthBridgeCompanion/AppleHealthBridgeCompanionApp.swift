import SwiftUI

@main
struct AppleHealthBridgeCompanionApp: App {
    @StateObject private var viewModel = SyncViewModel(
        bridgeClient: BridgeClient(),
        healthKitWriter: HealthKitWriter()
    )

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
        }
    }
}

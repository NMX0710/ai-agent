import Foundation

@MainActor
final class SyncViewModel: ObservableObject {
    @Published var configuration = BridgeConfiguration()
    @Published var lastSummary: SyncSummary?
    @Published var isSyncing = false
    @Published var isTestingConnection = false
    @Published private(set) var authorizationStatusText: String

    private let bridgeClient: BridgeClient
    private let healthKitWriter: HealthKitWriter

    init(bridgeClient: BridgeClient, healthKitWriter: HealthKitWriter) {
        self.bridgeClient = bridgeClient
        self.healthKitWriter = healthKitWriter
        self.authorizationStatusText = healthKitWriter.authorizationStatusDescription()
    }

    func testConnection() async {
        isTestingConnection = true
        defer { isTestingConnection = false }

        do {
            let response = try await bridgeClient.ping(configuration: configuration)
            lastSummary = SyncSummary(
                title: "Connection OK",
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: "Backend responded to /ping with status=\(response.status).",
                timestamp: Date()
            )
        } catch {
            lastSummary = SyncSummary(
                title: "Connection failed",
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: error.localizedDescription,
                timestamp: Date()
            )
        }
    }

    func requestAuthorization() async {
        do {
            try await healthKitWriter.requestAuthorization()
            authorizationStatusText = healthKitWriter.authorizationStatusDescription()
            lastSummary = SyncSummary(
                title: "Health access granted",
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: "The app can now write nutrition samples to Apple Health.",
                timestamp: Date()
            )
        } catch {
            lastSummary = SyncSummary(
                title: "Authorization failed",
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: error.localizedDescription,
                timestamp: Date()
            )
        }
    }

    func syncNow() async {
        guard !configuration.normalizedUserID.isEmpty else {
            lastSummary = SyncSummary(
                title: "Missing user ID",
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: "Set the backend user ID before syncing.",
                timestamp: Date()
            )
            return
        }

        isSyncing = true
        defer { isSyncing = false }

        do {
            try await healthKitWriter.requestAuthorization()
            authorizationStatusText = healthKitWriter.authorizationStatusDescription()

            let tasks = try await bridgeClient.fetchPendingWrites(configuration: configuration)
            if tasks.isEmpty {
                lastSummary = SyncSummary(
                    title: "Nothing to sync",
                    processed: 0,
                    succeeded: 0,
                    failed: 0,
                    message: "No pending Apple Health writes were returned by the backend.",
                    timestamp: Date()
                )
                return
            }

            var succeeded = 0
            var failed = 0
            var messages: [String] = []

            for task in tasks {
                do {
                    let externalID = try await healthKitWriter.write(task: task)
                    let response = try await bridgeClient.reportWriteResult(
                        configuration: configuration,
                        result: WriteResultRequest(
                            userID: task.userID,
                            draftID: task.draftID,
                            success: true,
                            claimToken: task.claimToken,
                            externalID: externalID,
                            error: nil
                        )
                    )
                    succeeded += 1
                    messages.append("\(task.draftID): \(response.status ?? "synced")")
                } catch {
                    failed += 1
                    _ = try? await bridgeClient.reportWriteResult(
                        configuration: configuration,
                        result: WriteResultRequest(
                            userID: task.userID,
                            draftID: task.draftID,
                            success: false,
                            claimToken: task.claimToken,
                            externalID: nil,
                            error: error.localizedDescription
                        )
                    )
                    messages.append("\(task.draftID): \(error.localizedDescription)")
                }
            }

            lastSummary = SyncSummary(
                title: failed == 0 ? "Sync completed" : "Sync completed with errors",
                processed: tasks.count,
                succeeded: succeeded,
                failed: failed,
                message: messages.joined(separator: "\n"),
                timestamp: Date()
            )
        } catch {
            lastSummary = SyncSummary(
                title: "Sync failed",
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: error.localizedDescription,
                timestamp: Date()
            )
        }
    }
}

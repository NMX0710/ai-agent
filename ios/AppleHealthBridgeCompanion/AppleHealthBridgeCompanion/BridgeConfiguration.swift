import Foundation

struct BridgeConfiguration {
    private enum Keys {
        static let baseURL = "bridge.base_url"
        static let bridgeToken = "bridge.token"
        static let userID = "bridge.user_id"
    }

    var baseURL: String {
        didSet { UserDefaults.standard.set(baseURL, forKey: Keys.baseURL) }
    }

    var bridgeToken: String {
        didSet { UserDefaults.standard.set(bridgeToken, forKey: Keys.bridgeToken) }
    }

    var userID: String {
        didSet { UserDefaults.standard.set(userID, forKey: Keys.userID) }
    }

    init() {
        self.baseURL = UserDefaults.standard.string(forKey: Keys.baseURL) ?? "http://127.0.0.1:8000"
        self.bridgeToken = UserDefaults.standard.string(forKey: Keys.bridgeToken) ?? ""
        self.userID = UserDefaults.standard.string(forKey: Keys.userID) ?? ""
    }

    var normalizedBaseURL: URL? {
        let value = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        return URL(string: value)
    }
}

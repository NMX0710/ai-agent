import Foundation

enum BridgeClientError: LocalizedError {
    case invalidBaseURL
    case invalidResponse
    case serverError(String)
    case transportError(String)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Backend base URL is invalid."
        case .invalidResponse:
            return "Backend returned an invalid response."
        case .serverError(let message):
            return message
        case .transportError(let message):
            return message
        }
    }
}

final class BridgeClient {
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private static let iso8601FormatterWithFractionalSeconds: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
    private static let iso8601Formatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            if let date = Self.iso8601FormatterWithFractionalSeconds.date(from: value) {
                return date
            }
            if let date = Self.iso8601Formatter.date(from: value) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Expected ISO8601 date string, got \(value)"
            )
        }
        self.encoder = JSONEncoder()
    }

    func ping(configuration: BridgeConfiguration) async throws -> PingResponse {
        guard let baseURL = configuration.normalizedBaseURL else {
            throw BridgeClientError.invalidBaseURL
        }

        let request = try makeRequest(
            url: baseURL.appending(path: "/ping"),
            bridgeToken: configuration.bridgeToken,
            jsonBody: nil
        )

        let (responseData, response) = try await perform(request)
        try validate(response: response, data: responseData)
        return try decoder.decode(PingResponse.self, from: responseData)
    }

    func fetchPendingWrites(configuration: BridgeConfiguration, limit: Int = 20, leaseSeconds: Int = 120) async throws -> [PendingWriteTask] {
        guard let baseURL = configuration.normalizedBaseURL else {
            throw BridgeClientError.invalidBaseURL
        }

        let body = [
            "user_id": configuration.normalizedUserID,
            "limit": limit,
            "lease_seconds": leaseSeconds
        ] as [String : Any]

        let data = try JSONSerialization.data(withJSONObject: body)
        let request = try makeRequest(
            url: baseURL.appending(path: "/integrations/apple-health/pending-writes"),
            bridgeToken: configuration.normalizedBridgeToken,
            jsonBody: data
        )

        let (responseData, response) = try await perform(request)
        try validate(response: response, data: responseData)
        return try decoder.decode(PendingWritesResponse.self, from: responseData).items
    }

    func reportWriteResult(configuration: BridgeConfiguration, result: WriteResultRequest) async throws -> WriteResultResponse {
        guard let baseURL = configuration.normalizedBaseURL else {
            throw BridgeClientError.invalidBaseURL
        }

        let request = try makeRequest(
            url: baseURL.appending(path: "/integrations/apple-health/write-result"),
            bridgeToken: configuration.normalizedBridgeToken,
            jsonBody: try encoder.encode(result)
        )

        let (responseData, response) = try await perform(request)
        try validate(response: response, data: responseData)
        return try decoder.decode(WriteResultResponse.self, from: responseData)
    }

    private func makeRequest(url: URL, bridgeToken: String, jsonBody: Data?) throws -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = jsonBody == nil ? "GET" : "POST"
        request.httpBody = jsonBody
        if jsonBody != nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        if !bridgeToken.isEmpty {
            request.setValue(bridgeToken, forHTTPHeaderField: "X-Apple-Bridge-Token")
        }
        request.timeoutInterval = 15
        return request
    }

    private func perform(_ request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await session.data(for: request)
        } catch {
            throw BridgeClientError.transportError(Self.describe(error: error, url: request.url))
        }
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw BridgeClientError.invalidResponse
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw BridgeClientError.serverError(message)
        }
    }

    private static func describe(error: Error, url: URL?) -> String {
        let endpoint = url?.absoluteString ?? "unknown URL"

        if let urlError = error as? URLError {
            switch urlError.code {
            case .notConnectedToInternet:
                return "No internet connection while reaching \(endpoint)."
            case .timedOut:
                return "Request to \(endpoint) timed out."
            case .cannotFindHost:
                return "Cannot find host for \(endpoint)."
            case .cannotConnectToHost:
                return "Cannot connect to host for \(endpoint). Check that the backend is reachable from the iPhone."
            case .networkConnectionLost:
                return "Network connection was lost while reaching \(endpoint)."
            case .secureConnectionFailed:
                return "Secure connection failed for \(endpoint)."
            case .appTransportSecurityRequiresSecureConnection:
                return "App Transport Security blocked \(endpoint)."
            default:
                return "Network error for \(endpoint): \(urlError.localizedDescription)"
            }
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            return "Network error for \(endpoint): \(nsError.localizedDescription)"
        }
        return "Request failed for \(endpoint): \(error.localizedDescription)"
    }
}

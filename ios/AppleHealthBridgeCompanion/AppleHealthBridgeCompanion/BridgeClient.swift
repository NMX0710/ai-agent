import Foundation

enum BridgeClientError: LocalizedError {
    case invalidBaseURL
    case invalidResponse
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Backend base URL is invalid."
        case .invalidResponse:
            return "Backend returned an invalid response."
        case .serverError(let message):
            return message
        }
    }
}

final class BridgeClient {
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .iso8601
        self.encoder = JSONEncoder()
    }

    func fetchPendingWrites(configuration: BridgeConfiguration, limit: Int = 20, leaseSeconds: Int = 120) async throws -> [PendingWriteTask] {
        guard let baseURL = configuration.normalizedBaseURL else {
            throw BridgeClientError.invalidBaseURL
        }

        let body = [
            "user_id": configuration.userID,
            "limit": limit,
            "lease_seconds": leaseSeconds
        ] as [String : Any]

        let data = try JSONSerialization.data(withJSONObject: body)
        let request = try makeRequest(
            url: baseURL.appending(path: "/integrations/apple-health/pending-writes"),
            bridgeToken: configuration.bridgeToken,
            jsonBody: data
        )

        let (responseData, response) = try await session.data(for: request)
        try validate(response: response, data: responseData)
        return try decoder.decode(PendingWritesResponse.self, from: responseData).items
    }

    func reportWriteResult(configuration: BridgeConfiguration, result: WriteResultRequest) async throws -> WriteResultResponse {
        guard let baseURL = configuration.normalizedBaseURL else {
            throw BridgeClientError.invalidBaseURL
        }

        let request = try makeRequest(
            url: baseURL.appending(path: "/integrations/apple-health/write-result"),
            bridgeToken: configuration.bridgeToken,
            jsonBody: try encoder.encode(result)
        )

        let (responseData, response) = try await session.data(for: request)
        try validate(response: response, data: responseData)
        return try decoder.decode(WriteResultResponse.self, from: responseData)
    }

    private func makeRequest(url: URL, bridgeToken: String, jsonBody: Data) throws -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = jsonBody
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !bridgeToken.isEmpty {
            request.setValue(bridgeToken, forHTTPHeaderField: "X-Apple-Bridge-Token")
        }
        return request
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
}

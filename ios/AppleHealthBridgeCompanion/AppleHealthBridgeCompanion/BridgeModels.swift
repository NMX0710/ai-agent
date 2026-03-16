import Foundation

struct PendingWritesResponse: Decodable {
    let ok: Bool
    let count: Int
    let items: [PendingWriteTask]
}

struct PendingWriteTask: Decodable, Identifiable {
    let draftID: String
    let userID: String
    let claimToken: String
    let payload: AppleHealthPayload

    var id: String { draftID }

    enum CodingKeys: String, CodingKey {
        case draftID = "draft_id"
        case userID = "user_id"
        case claimToken = "claim_token"
        case payload
    }
}

struct AppleHealthPayload: Decodable {
    let consumedAt: Date
    let timezone: String
    let mealType: String
    let foodType: String?
    let syncIdentifier: String
    let syncVersion: Int
    let samples: [NutritionSample]

    enum CodingKeys: String, CodingKey {
        case consumedAt = "consumed_at"
        case timezone
        case mealType = "meal_type"
        case foodType = "food_type"
        case syncIdentifier = "sync_identifier"
        case syncVersion = "sync_version"
        case samples
    }
}

struct NutritionSample: Decodable {
    let identifier: String
    let value: Double
    let unit: String
}

struct WriteResultRequest: Encodable {
    let userID: String
    let draftID: String
    let success: Bool
    let claimToken: String
    let externalID: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case userID = "user_id"
        case draftID = "draft_id"
        case success
        case claimToken = "claim_token"
        case externalID = "external_id"
        case error
    }
}

struct WriteResultResponse: Decodable {
    let ok: Bool
    let status: String?
    let error: String?
}

struct PingResponse: Decodable {
    let status: String
}

struct SyncSummary {
    let title: String
    let processed: Int
    let succeeded: Int
    let failed: Int
    let message: String
    let timestamp: Date
}

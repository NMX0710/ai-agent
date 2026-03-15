import Foundation
import HealthKit

enum HealthKitWriterError: LocalizedError {
    case notAvailable
    case unsupportedSample(String)
    case invalidDate
    case nothingToWrite

    var errorDescription: String? {
        switch self {
        case .notAvailable:
            return "HealthKit is not available on this device."
        case .unsupportedSample(let identifier):
            return "Unsupported HealthKit identifier: \(identifier)"
        case .invalidDate:
            return "Meal timestamp is invalid."
        case .nothingToWrite:
            return "No supported nutrition samples were provided."
        }
    }
}

final class HealthKitWriter {
    private let healthStore = HKHealthStore()

    func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthKitWriterError.notAvailable
        }

        let shareTypes = Set(Self.supportedIdentifiers.compactMap {
            HKObjectType.quantityType(forIdentifier: $0)
        })

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            healthStore.requestAuthorization(toShare: shareTypes, read: []) { success, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                if success {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: HealthKitWriterError.notAvailable)
                }
            }
        }
    }

    func authorizationStatusDescription() -> String {
        guard HKHealthStore.isHealthDataAvailable() else {
            return "Unavailable"
        }
        guard let energyType = HKObjectType.quantityType(forIdentifier: .dietaryEnergyConsumed) else {
            return "Unknown"
        }
        switch healthStore.authorizationStatus(for: energyType) {
        case .notDetermined:
            return "Not Requested"
        case .sharingDenied:
            return "Denied"
        case .sharingAuthorized:
            return "Authorized"
        @unknown default:
            return "Unknown"
        }
    }

    func write(task: PendingWriteTask) async throws -> String {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthKitWriterError.notAvailable
        }

        let objects = try makeObjects(from: task.payload)
        guard !objects.isEmpty else {
            throw HealthKitWriterError.nothingToWrite
        }

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
            healthStore.save(objects) { success, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                if success {
                    continuation.resume(returning: task.payload.syncIdentifier)
                } else {
                    continuation.resume(throwing: HealthKitWriterError.notAvailable)
                }
            }
        }
    }

    private func makeObjects(from payload: AppleHealthPayload) throws -> [HKObject] {
        let consumedAt = payload.consumedAt
        let metadata = buildMetadata(from: payload)

        return try payload.samples.map { sample in
            let typeIdentifier = HKQuantityTypeIdentifier(rawValue: sample.identifier)
            guard let quantityType = HKObjectType.quantityType(forIdentifier: typeIdentifier) else {
                throw HealthKitWriterError.unsupportedSample(sample.identifier)
            }
            let quantity = HKQuantity(unit: try unit(from: sample.unit), doubleValue: sample.value)
            return HKQuantitySample(
                type: quantityType,
                quantity: quantity,
                start: consumedAt,
                end: consumedAt,
                metadata: metadata
            )
        }
    }

    private func buildMetadata(from payload: AppleHealthPayload) -> [String: Any] {
        var metadata: [String: Any] = [
            HKMetadataKeySyncIdentifier: payload.syncIdentifier,
            HKMetadataKeySyncVersion: payload.syncVersion,
        ]
        if let foodType = payload.foodType, !foodType.isEmpty {
            metadata[HKMetadataKeyFoodType] = foodType
        }
        return metadata
    }

    private func unit(from rawUnit: String) throws -> HKUnit {
        switch rawUnit.lowercased() {
        case "kcal":
            return .kilocalorie()
        case "g":
            return .gram()
        case "ml":
            return .literUnit(with: .milli)
        default:
            throw HealthKitWriterError.unsupportedSample(rawUnit)
        }
    }

    private static let supportedIdentifiers: [HKQuantityTypeIdentifier] = [
        .dietaryEnergyConsumed,
        .dietaryProtein,
        .dietaryCarbohydrates,
        .dietaryFatTotal,
        .dietaryFiber,
        .dietarySugar,
        .dietarySodium,
        .dietaryCholesterol,
        .dietaryWater,
        .dietaryCalcium,
        .dietaryIron,
        .dietaryPotassium,
        .dietaryVitaminC,
        .dietaryVitaminD,
        .dietaryVitaminB12,
        .dietaryVitaminE,
    ]
}

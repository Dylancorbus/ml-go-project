/*
 * Finnhub API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package finnhub
// IndicesConstituents struct for IndicesConstituents
type IndicesConstituents struct {
	// Index's symbol.
	Symbol string `json:"symbol,omitempty"`
	// Array of constituents.
	Constituents []string `json:"constituents,omitempty"`
}

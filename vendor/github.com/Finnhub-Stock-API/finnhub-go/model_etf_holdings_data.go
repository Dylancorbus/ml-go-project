/*
 * Finnhub API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package finnhub
// EtfHoldingsData struct for EtfHoldingsData
type EtfHoldingsData struct {
	// Symbol description
	Symbol string `json:"symbol,omitempty"`
	// Number of shares owned by the ETF.
	Share float32 `json:"share,omitempty"`
	// Portfolio's percent
	Percent float32 `json:"percent,omitempty"`
}
